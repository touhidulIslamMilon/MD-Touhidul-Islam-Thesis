
import math
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import Trainer
from transformers import TrainingArguments
from accelerate import Accelerator
from transformers import get_scheduler

import torch
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def training_function(model,optimizer, tokenizer, chunk_size, batch_size, wwm_probability, data_collator, model_name, downsampled_dataset, test_dataset,eval_dataset,week):
    result = {}

    # Load model with eager attention implementation
    model = AutoModelForSequenceClassification.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare data loaders
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # Initialize the accelerator
    accelerator = Accelerator()
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Set up the learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    output_dir = model_name
    print("Device: ", accelerator.state.device)  # Use accelerator to get device

    # Training and Evaluation Loop
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            # Move batch to the appropriate device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        eval_losses = []
        for step, batch in enumerate(test_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}  # Move to device
            
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            eval_losses.append(accelerator.gather(loss.repeat(batch_size)))

        eval_losses = torch.cat(eval_losses)
        eval_losses = eval_losses[: len(test_dataset)]

        # Log evaluation loss or metrics here if needed

        # Save and upload model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
        
        # Optional: Log to Hugging Face Hub
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
    
    return result


# batch_size = 64
# Show the training loss with every epoch
def training_function2(model, tokenizer, chunk_size, batch_size, wwm_probability, data_collator, model_name, downsampled_dataset):
    logging_steps = len(downsampled_dataset["train"]) // batch_size



# and move our model over to the selected device
    model.to(device)
# activate training mode
    model.train()

    training_args = TrainingArguments(
    output_dir=f"{model_name}-retrain-week",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    batch_size=batch_size,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #push_to_hub=True,
    #fp16=True,
    logging_steps=logging_steps,    
    )


    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    )


    eval_results = trainer.evaluate()
    print(f">>> Before Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.train()


    eval_results = trainer.evaluate()
    print(f">>> After Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

