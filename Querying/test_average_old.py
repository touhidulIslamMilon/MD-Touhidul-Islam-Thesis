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

# Download NLTK resources
nltk.download('punkt')

# Set environment variable for transformers cache
os.environ['TRANSFORMERS_CACHE'] = '/pfs/work7/workspace/scratch/ma_mislam-newthesis/ma_mislam-mislam_twitterlm_thesis/cache'
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
        
        week_results = process_single_model_all_prompts(week, tokenizer, options, prompts, epoch_num)
        if week_results is None:
            print(f"Skipping week {week} due to error")
            continue
            
        for prompt in prompts:
            scores = week_results[prompt]
            
            # Add the week's scores as a new column
            week_scores = []
            for emotion in options:
                week_scores.append(scores.get(emotion, 0.0))
            all_results[prompt][week] = week_scores
            
    # Format all results
    for prompt in prompts:
        all_results[prompt].index.name = 'Emotion'
        all_results[prompt].columns = pd.to_datetime(
            [f"{col}_4" for col in all_results[prompt].columns],
            format="%Y_%W_%w")
    
    return list(all_results.values())

options = ["Sad", "Happy", "Content", "Stressed", "Lonely", "Inspired", 
           "Frustrated", "Optim@@", "Bored", "A@@", "Ener@@"]
def process_single_model_all_prompts(week, tokenizer, options, prompts, epoch_num):
    """Process all prompts with a single model"""
    try:
        model_name = f"/pfs/work7/workspace/scratch/ma_mislam-newthesis/ma_mislam-Thesis_Touhidul/new1/training_batch_new/BERTweet_retrain_{week}"
        model_name = getcheckpoint(model_name, epoch_num)
        
        # Create pipeline with original targets
        pipe = pipeline("fill-mask", 
                       tokenizer=tokenizer,
                       model=model_name,
                       top_k=100,
                       targets=options)  # Keep the targets
        
        week_results = {}
        print(f"\nProcessing Week: {week}")
        
        for prompt_idx, prompt in enumerate(prompts):
            output = pipe(prompt)
            scores = {emotion: 0.0 for emotion in options}
            
            print(f"\nPrompt {prompt_idx + 1}:")
            print(f"Raw output from model (length: {len(output)}):")
            for result in output:
                token = result['token_str'].strip()
                score = float(result['score'])
                print(f"Token: {token}, Score: {score}")
                
                # Direct mapping for special tokens
                if token == 'Optim@@':
                    scores['Optimistic'] = score
                elif token == 'A@@':
                    scores['Apathetic'] = score
                elif token == 'Ener@@':
                    scores['Energetic'] = score
                else:
                    # For other tokens, use them directly as they match the options
                    for option in options:
                        if token == option:
                            scores[option] = score
            
            week_results[prompt] = scores
            
            print("\nFinal scores:")
            for emotion, score in scores.items():
                print(f"{emotion}: {score}")
            
        return week_results
        
    except Exception as e:
        print(f"Error processing week {week}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_results_sequential(prompts, weeks, tokenizer, options):
    """Process results sequentially"""
    epoch_num = 2
    
    # Initialize DataFrames with proper structure
    all_results = {}
    for prompt in prompts:
        df = pd.DataFrame(index=options)
        all_results[prompt] = df
    
    for week in tqdm(weeks, desc="Processing weeks"):
        week_results = process_single_model_all_prompts(week, tokenizer, options, prompts, epoch_num)
        if week_results is None:
            print(f"Skipping week {week} due to error")
            continue
            
        for prompt in prompts:
            scores = week_results[prompt]
            all_results[prompt][week] = pd.Series(scores)
    
    # Format results
    for prompt in prompts:
        df = all_results[prompt]
        df.index.name = 'Emotion'
        df.columns = pd.to_datetime([f"{col}_4" for col in df.columns], format="%Y_%W_%w")
        
        print(f"\nResults for prompt: {prompt[:100]}...")
        print(df)
    
    return list(all_results.values())

# Define the options with the correct tokens



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
        token_mapping = {
            'Optim@@': 'Optimistic',
            'A@@': 'Apathetic',
            'Ener@@': 'Energetic'
        }
        result.rename(index=token_mapping, inplace=True)
        
        return result
        
    except Exception as e:
        print(f"Error in get_average: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
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
# Define prompts and options
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

options = ["Sad", "Happy", "Content", "Stressed", "Lonely", "Inspired", 
          "Frustrated", "Optimistic", "Bored", "Apathetic", "Energetic"]
def main():
    # Initialize
    epoch_num = 2
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    years = [2019, 2020, 2021, 2022,2023]
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
        results.to_csv(f"intermediate_results/prompt_{i}_rawaverageold_results.csv")
    
    # Calculate and save different averages
    results_single = get_average(result_list, 'single')
    results_single.to_csv(f"ten/withoutsim/Single/oldaveragesingle_query_epoch{epoch_num}.csv")
    
    results_average = get_average(result_list, 'average')
    results_average.to_csv(f"ten/withoutsim/Mean/oldaverageMulti_mean_query_epoch{epoch_num}.csv")
    
    results_median = get_average(result_list, 'median')
    results_median.to_csv(f"ten/withoutsim/Median/oldaverageMulti_median_query_epoch{epoch_num}.csv")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()