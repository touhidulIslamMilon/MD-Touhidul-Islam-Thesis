from transformers import pipeline, AutoTokenizer
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm


# Set environment variable for transformers cache
os.environ['TRANSFORMERS_CACHE'] = '/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/cache'

from sklearn.metrics.pairwise import cosine_similarity



prompts= [
"Broadly speaking, which of the following best describes your mood and/or how you have felt in the past week? Please select all that apply. I am feeling <mask> today.",
"In consideration of your overall emotional state and experiences over the past week, please indicate all options that accurately reflect how you're feeling today <mask>.",
"Reflecting on the various emotions you've experienced recently, which statements resonate with how you're feeling today <mask>?",
"When thinking about your mood and feelings over the past week, which descriptions closely match your current feelings today <mask>?",
"Please select all the options that best describe your current mood and emotions, based on your experiences over the past week <mask>."
"From your mood and emotions in the past week, which statements accurately reflect your current feelings today <mask>?",
"Reflecting on your mood and emotions over the past week, which descriptions most closely match your current feelings today <mask>?",
"Given your mood and emotions over the past week, identify all that accurately reflect how you're feeling today <mask>.",
"Tick all the statements that align with your current mood and feelings, considering your experiences over the past week <mask>.",
"Considering your overall mood and emotions from the past week, please choose all that apply to how you're feeling today <mask>.",
]
options = ["Sad", "Happy", "Content", "Other", "Stressed", "Lonely", "Inspired", "Frustrated", "Optim@@", "Bored", "A@@", "Ener@@"]

def getWeek(years):
    weeks = []
    for year in years:
        if year == 2019:
            for i in range(22, 53):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
        elif year == 2023:
            for i in range(1, 10):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
        else:
            for i in range(1, 53):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
    print(f"Generated {len(weeks)} weeks")
    return weeks
def categorize_emotion(word):
    ps = PorterStemmer()
    
    emotion_categories_custom = {
        "Sad": ["sad", "downhearted", "blue", "unhappy", "melancholy"],
        "Happy": ["cheerful", "excited", "joyful", "jovial", "elated"],
        "Content": ["content", "satisfied", "peaceful", "relaxed"],
        "Stressed": ["stressed", "anxious", "tense", "nervous"],
        "Lonely": ["lonely", "isolated", "alone", "abandoned"],
        "Inspired": ["inspired", "motivated", "creative", "uplifted"],
        "Frustrated": ["frustrated", "angry", "annoyed", "irritated"],
        "Optimistic": ["hopeful", "optimistic", "positive", "confident"],
        "Bored": ["bored", "apathetic", "uninterested", "disengaged"],
        "Apathetic": ["apathetic", "indifferent", "unconcerned"],
        "Energetic": ["energetic", "active", "vibrant", "lively"]
    }

    word = word.lower()
    word_stemmed = ps.stem(word)
    
    for category, words in emotion_categories_custom.items():
        if any(word_stemmed == ps.stem(w.lower()) for w in words):
            return category
    return "Other"

def calculate_results_sequential(prompts, weeks, tokenizer, options):
    """Process one model at a time for all prompts"""
    epoch_num = 2
    
    # Initialize DataFrames with proper structure
    all_results = {}
    for prompt in prompts:
        # Create DataFrame with proper index from the start
        df = pd.DataFrame(index=options)
        all_results[prompt] = df
    
    for week in tqdm(weeks, desc="Processing weeks"):
        print(f"\nProcessing week: {week}")
        
        week_results = process_single_model_all_prompts_with_similarity(week, tokenizer, options, prompts, epoch_num)
        if week_results is None:
            print(f"Skipping week {week} due to error")
            continue
            
        for prompt in prompts:
            
            scores = week_results[prompt]
            # Add the week's scores as a new column
            week_scores = []
            for emotion in options:
                week_scores.append(scores.get(emotion, 0.0))
            else:
                print("Not in option")
                print(emotion)
                print(options)
            all_results[prompt][week] = week_scores
            print("---------")
            print(week_scores)
         
    # Format all results
    for prompt in prompts:
        all_results[prompt].index.name = 'Emotion'
        all_results[prompt].columns = pd.to_datetime(
            [f"{col}_4" for col in all_results[prompt].columns],
            format="%Y_%W_%w")
   
    return list(all_results.values())
def calculate_emotion_similarity(word, target_emotion, emotion_categories_panas_x, emotion_categories_custom):
    """Calculate similarity based on emotion categories"""
    ps = PorterStemmer()
    word = word.lower()
    word_stemmed = ps.stem(word)
    
    # Get categories for the input word
    word_panas, word_custom = categorize_emotion(word)
    target_panas, target_custom = categorize_emotion(target_emotion)
    
    similarity_score = 0.0
    
    # Check direct category matches
    if word_custom == target_custom and word_custom != "Not in the list":
        similarity_score = 1.0
    elif word_panas == target_panas and word_panas != "Not in the list":
        similarity_score = 0.8  # Slightly lower weight for PANAS-X matches
    else:
        # Check if words appear in same category lists
        for category, words in emotion_categories_custom.items():
            word_in_category = any(ps.stem(w.lower()) == word_stemmed for w in words)
            target_in_category = any(ps.stem(w.lower()) == ps.stem(target_emotion.lower()) for w in words)
            if word_in_category and target_in_category:
                similarity_score = max(similarity_score, 0.9)
                
        for category, words in emotion_categories_panas_x.items():
            word_in_category = any(ps.stem(w.lower()) == word_stemmed for w in words)
            target_in_category = any(ps.stem(w.lower()) == ps.stem(target_emotion.lower()) for w in words)
            if word_in_category and target_in_category:
                similarity_score = max(similarity_score, 0.7)
    
    return similarity_score
def getcheckpoint(checkpoint_dir, epoch_num): 
    # Step 1: List all directories (checkpoints) in the directory
    available_checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    # Step 2: Sort the checkpoints by the epoch number (assuming they follow the pattern 'checkpoint-epoch-X')
    # If your checkpoints follow a pattern like 'checkpoint-epoch-1', 'checkpoint-epoch-2', etc.
    available_checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sorting by the last part after 'epoch-'
    
    
    # Step 3: Load the latest checkpoint (or any specific checkpoint you want)
    # Select the latest checkpoint (the last one in the sorted list)
    if epoch_num == 1:
        checkpoint = available_checkpoints[-1] 
    elif epoch_num == 2:
        checkpoint = available_checkpoints[1]
    else:
        checkpoint = available_checkpoints[2]
    model_dir = checkpoint_dir+'/'+checkpoint
    print(f"Loading the latest checkpoint: {model_dir}")
    return model_dir
def process_single_model_all_prompts_with_similarity(week, tokenizer, options, prompts, epoch_num):
    """Process all prompts with emotion-based similarity scoring"""
    try:
        model_name = f"/pfs/work7/workspace/scratch/ma_mislam-newthesis/ma_mislam-Thesis_Touhidul/new1/training_batch_new/BERTweet_retrain_{week}"
        model_name = getcheckpoint(model_name, epoch_num)
        
        # Define emotion categories with basic words that match your options
        emotion_categories_custom = {
            "Sad": [
                "sad", "down", "blue", "unhappy", "gloomy", "miserable", 
                "heartbroken", "depressed", "sorrowful"
            ],
            "Happy": [
                "happy", "excited", "joyful", "glad", "cheerful", "delighted",
                "joy", "pleased", "thrilled", "merry", "elated"
            ],
            "Content": [
                "content", "satisfied", "peaceful", "relaxed", "calm", 
                "serene", "at ease", "comfortable", "tranquil", "settled"
            ],
            "Stressed": [
                "stressed", "anxious", "tense", "nervous", "worried", 
                "uneasy", "afraid", "scared", "fearful", "distressed",
                "overwhelmed", "panic", "restless"
            ],
            "Lonely": [
                "lonely", "alone", "isolated", "solitary", "abandoned",
                "disconnected", "separated", "remote", "detached"
            ],
            "Inspired": [
                "inspired", "motivated", "creative", "enthusiastic", 
                "driven", "stimulated", "encouraged", "uplifted",
                "imaginative", "innovative"
            ],
            "Frustrated": [
                "frustrated", "angry", "annoyed", "mad", "irritated",
                "agitated", "hostile", "upset", "furious", "outraged",
                "bitter", "resentful", "irate"
            ],
            "Optimistic": [  # Maps to Optim@@
                "hopeful", "positive", "confident", "optimistic", 
                "promising", "bright", "assured", "strong", "proud",
                "bold", "determined", "certain"
            ],
            "Bored": [
                "bored", "tired", "dull", "weary", "sleepy", "exhausted",
                "fatigued", "uninterested", "listless", "monotonous",
                "tedious", "drowsy", "sluggish"
            ],
            "Apathetic": [  # Maps to A@@
                "apathy", "cold", "numb", "indifferent", "apathetic",
                "unfeeling", "disinterested", "detached", "distant",
                "emotionless", "uncaring", "unmoved"
            ],
            "Energetic": [  # Maps to Ener@@
                "energetic", "active", "lively", "vibrant", "dynamic",
                "vigorous", "spirited", "animated", "peppy", "alert",
                "enthusiastic", "invigorated", "vital"
            ]
        }
        
        emotion_categories_panas_x = {
            "Happy": ["joy", "cheerful", "happy", "excited", "delighted"],
            "Self-Assurance": ["confident", "bold", "strong", "proud", "optimistic"],
            "Attentiveness": ["alert", "focused", "aware", "energetic"],
            "Serenity": ["calm", "peaceful", "relaxed", "content"],
            "Surprise": ["surprised", "amazed", "shocked"],
            "Fear": ["scared", "afraid", "nervous", "anxious"],
            "Hostility": ["angry", "hostile", "annoyed", "mad"],
            "Guilt": ["guilty", "ashamed", "regret"],
            "Sadness": ["sad", "down", "blue", "unhappy"],
            "Shyness": ["shy", "quiet"],
            "Fatigue": ["tired", "sleepy", "exhausted", "weary"]
        }
        
        # Define special token mappings
        special_token_mapping = {
            "Optimistic": "Optim@@",
            "Apathetic": "A@@",
            "Energetic": "Ener@@"
        }
        
        # Replace special token categories in options with their actual tokens
        modified_options = []
        for option in options:
            if option in special_token_mapping:
                modified_options.append(special_token_mapping[option])
            else:
                modified_options.append(option)
        
        # Get valid target words
        target_words = get_valid_emotion_words(tokenizer, emotion_categories_panas_x, emotion_categories_custom)
        print(f"\nProcessing Week: {week}")
        print(f"Using {len(target_words)} valid target words: {target_words}")
        
        # Create pipeline with valid target words
        pipe = pipeline("fill-mask", 
                       tokenizer=tokenizer,
                       model=model_name,
                       targets=target_words,
                       top_k=len(target_words))
        
        week_results = {}
        
        for prompt_idx, prompt in enumerate(prompts):
            output = pipe(prompt)
            
            # Initialize scores with modified options
            scores = {option: 0.0 for option in modified_options}
            print(f"Initialized scores with options: {list(scores.keys())}")
            
            print(f"\nPrompt {prompt_idx + 1} predictions:")
            for result in output:
                token = result['token_str'].strip()
                pred_score = float(result['score'])
                
                token_panas, token_custom = categorize_emotion(token, 
                                                            emotion_categories_panas_x, 
                                                            emotion_categories_custom)

                
                # Update scores based on custom categorization
                if token_custom != "Not in custom":
                    # Map category to special token if needed
                    option_name = special_token_mapping.get(token_custom, token_custom)
                    if option_name in scores:
                        scores[option_name] += pred_score
                        #print(f"Token: {token:<15} Mapped to: {option_name:<15} Score: {pred_score:.4f}")
                    else:
                        print(f"Warning: {option_name} not in scores dictionary. Available options: {list(scores.keys())}")
                
                # Consider PANAS-X categorization with lower weight
                if token_panas != "Not in PANAS-X":
                    panas_to_option = {
                        "Happy": "Happy",
                        "Sadness": "Sad",
                        "Fear": "Stressed",
                        "Serenity": "Content",
                        "Fatigue": "Bored",
                        "Self-Assurance": "Optim@@",
                        "Attentiveness": "Ener@@"
                    }
                    if token_panas in panas_to_option:
                        option_name = panas_to_option[token_panas]
                        if option_name in scores:
                            scores[option_name] += pred_score * 0.8
                           # print(f"PANAS-X mapping: {token_panas} -> {option_name}")
                        else:
                            print(f"Warning: PANAS-X mapping {option_name} not in scores dictionary")
            
            # Scale up scores
            for emotion in scores:
                scores[emotion] *= 1
            
            week_results[prompt] = scores
            
            print(f"\nFinal scores for Prompt {prompt_idx + 1}:")
            for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0:
                    print(f"{emotion:<15}: {score:.4f}")
        #print(week_results)
        return week_results
        
    except Exception as e:
        print(f"Error processing week {week}: {str(e)}")
        traceback.print_exc()
        return None

def get_valid_emotion_words(tokenizer, emotion_categories_panas_x, emotion_categories_custom):
    """Get list of valid emotion words that exist in the tokenizer vocabulary"""
    # Flatten all emotion words from both dictionaries
    all_words = set()
    
    # Add words from PANAS-X categories
    for words in emotion_categories_panas_x.values():
        all_words.update(words)
    
    # Add words from custom categories
    for category, words in emotion_categories_custom.items():
        all_words.update(words)
    
    # Explicitly add special tokens
    special_tokens = ["Optim@@", "A@@", "Ener@@"]
    all_words.update(special_tokens)
    
    # Additional words for special token categories
    special_token_words = {
        # Optimistic (Optim@@) words
        "optimistic", "hopeful", "positive", "confident", "promising", 
        "bright", "assured", "strong", "proud", "bold", "determined",
        
        # Apathetic (A@@) words
        "apathetic", "apathy", "cold", "numb", "indifferent", "unfeeling",
        "disinterested", "detached", "distant", "emotionless",
        
        # Energetic (Ener@@) words
        "energetic", "active", "lively", "vibrant", "dynamic", "vigorous",
        "spirited", "animated", "peppy", "alert", "enthusiastic"
    }
    
    all_words.update(special_token_words)
    
    # Convert to list and sort for consistency
    all_words = sorted(list(all_words))
    
    # Filter words that exist in tokenizer vocabulary
    valid_words = []
    for word in all_words:
        # Special handling for special tokens
        if word in special_tokens:
            valid_words.append(word)
            continue
            
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id != tokenizer.unk_token_id:
            valid_words.append(word)
    
    print(f"Found {len(valid_words)} valid words out of {len(all_words)} total words")
    print(f"\nSpecial tokens included: {[w for w in valid_words if w in special_tokens]}")
    return valid_words
def categorize_emotion(word, emotion_categories_panas_x, emotion_categories_custom):
    """Categorize a word into both PANAS-X and custom emotion categories"""
    ps = PorterStemmer()
    word = word.lower()
    word_stemmed = ps.stem(word)
    
    # Find PANAS-X category
    panas_x_category = "Not in PANAS-X"
    for category, words in emotion_categories_panas_x.items():
        if any(ps.stem(w.lower()) == word_stemmed for w in words):
            panas_x_category = category
            break
    
    # Find custom category
    custom_category = "Not in custom"
    for category, words in emotion_categories_custom.items():
        if any(ps.stem(w.lower()) == word_stemmed for w in words):
            custom_category = category
            break
    
    return panas_x_category, custom_category


def get_average(dfs, selection_type):
    """Calculate average/median/single results from multiple dataframes"""
    try:
        # Print shape and sample of input dataframes
        print(f"\nNumber of dataframes: {len(dfs)}")
        for i, df in enumerate(dfs):
            print(f"\nDataFrame {i+1} shape: {df.shape}")
            print("Sample of values:")
            print(df.iloc[:5, :2])  # Show first 5 rows, 2 columns
        
        # Combine dataframes
        if selection_type == 'average':
            print("\nCalculating mean across all dataframes")
            # Stack all dataframes and calculate mean
            all_values = pd.concat([df for df in dfs], axis=1)
            result = all_values.groupby(level=0, axis=1).mean()
            
        elif selection_type == 'median':
            print("\nCalculating median across all dataframes")
            # Stack all dataframes and calculate median
            all_values = pd.concat([df for df in dfs], axis=1)
            result = all_values.groupby(level=0, axis=1).median()
            
        else:  # 'single'
            print("\nUsing single dataframe")
            result = dfs[0]
        
        # Print intermediate results
        print("\nResult shape:", result.shape)
        print("Sample of results:")
        print(result.iloc[:5, :2])
        
        # Ensure proper index naming
        result.index.name = 'Emotion'
        
        # Map special tokens back to original names

        
        return result
        
    except Exception as e:
        print(f"Error in get_average: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Define prompts and options
prompts = [
    "Broadly speaking, which of the following best describes your mood and/or how you have felt in the past week? Please select all that apply. I am feeling <mask> today.",
"In consideration of your overall emotional state and experiences over the past week, please indicate all options that accurately reflect how you're feeling today <mask>.",
"Reflecting on the various emotions you've experienced recently, which statements resonate with how you're feeling today <mask>?",
"When thinking about your mood and feelings over the past week, which descriptions closely match your current feelings today <mask>?",
"Please select all the options that best describe your current mood and emotions, based on your experiences over the past week <mask>."
"From your mood and emotions in the past week, which statements accurately reflect your current feelings today <mask>?",
"Reflecting on your mood and emotions over the past week, which descriptions most closely match your current feelings today <mask>?",
"Given your mood and emotions over the past week, identify all that accurately reflect how you're feeling today <mask>.",
"Tick all the statements that align with your current mood and feelings, considering your experiences over the past week <mask>.",
"Considering your overall mood and emotions from the past week, please choose all that apply to how you're feeling today <mask>.",
]

options =  ["Sad", "Happy", "Content", "Other", "Stressed", "Lonely", "Inspired", "Frustrated", "Optim@@", "Bored", "A@@", "Ener@@"]

def main():
    # Initialize
    epoch_num = 2
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    years = [2019, 2020, 2021, 2022, 2023]
    weeks = getWeek(years)
    #weeks= ['2019_22', '2019_23']
    # Create output directories if they don't exist
    os.makedirs("intermediate_results", exist_ok=True)
    os.makedirs("ten/withsim/Single", exist_ok=True)
    os.makedirs("ten/withsim/Mean", exist_ok=True)
    os.makedirs("ten/withsim/Median", exist_ok=True)
    
    print(f"Starting processing for {len(weeks)} weeks...")
    
    # Process all weeks sequentially
    result_list = calculate_results_sequential(prompts, weeks, tokenizer, options)
    
    # Save intermediate results
    for i, results in enumerate(result_list):
        results.to_csv(f"intermediate_results/prompt_{i}_rawwithsimold_results.csv")
    
    # Calculate and save different averages
    results_single = get_average(result_list, 'single')
    results_single.to_csv(f"ten/withsim/Single/single_query_epoch{epoch_num}.csv")
    
    results_average = get_average(result_list, 'average')
    results_average.to_csv(f"ten/withsim/Mean/Multi_mean_query_epoch{epoch_num}.csv")
    
    results_median = get_average(result_list, 'median')
    results_median.to_csv(f"ten/withsim/Median/Multi_median_query_epoch{epoch_num}.csv")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()