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
from nltk.corpus import wordnet

# Set environment variable for transformers cache
os.environ['TRANSFORMERS_CACHE'] = '/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/cache'

from sklearn.metrics.pairwise import cosine_similarity

prompts= [
"Broadly speaking, which of the following best describes your mood and/or how you have felt in the past week? Please select all that apply. I am feeling <mask> today.",
"In consideration of your overall emotional state and experiences over the past week, please indicate all options that accurately reflect how you're feeling today <mask>.",
"Reflecting on the various emotions you've experienced recently, which statements resonate with how you're feeling today <mask>?",
"When thinking about your mood and feelings over the past week, which descriptions closely match your current feelings today <mask>?",
"Please select all the options that best describe your current mood and emotions, based on your experiences over the past week <mask>.",
"From your mood and emotions in the past week, which statements accurately reflect your current feelings today <mask>?",
"Reflecting on your mood and emotions over the past week, which descriptions most closely match your current feelings today <mask>?",
"Given your mood and emotions over the past week, identify all that accurately reflect how you're feeling today <mask>.",
"Tick all the statements that align with your current mood and feelings, considering your experiences over the past week <mask>.",
"Considering your overall mood and emotions from the past week, please choose all that apply to how you're feeling today <mask>."
]
options = ["Sad", "Happy", "Content", "Stressed", "Lonely", "Inspired", "Frustrated", "Optimistic", "Bored", "Apathetic", "Energetic"]
def categorize_emotion(arg):
    check_argument_type(arg)
    for emotion, keywords in emotion_keywords.items():
        #print(emotion, keywords)
        if arg.lower() in [keyword.lower() for keyword in keywords]:  # Match the word case-insensitively
            return emotion
    return 'Unknown' 
emotion_keywords = {
    'Sad': ['sad', 'deplorable', 'pitiful', 'sorry', 'unhappy'],
    'Happy': ['happy', 'joyful', 'glad', 'cheerful'],
    'Content': ['content', 'satisfied', 'fulfilled', 'pleased'],
    'Stressed': ['stressed', 'anxious', 'overwhelmed', 'tense'],
    'Lonely': ['lonely', 'alone', 'isolated', 'solitary'],
    'Inspired': ['inspired', 'motivated', 'driven', 'empowered'],
    'Frustrated': ['frustrated', 'annoyed', 'irritated', 'disappointed'],
    'Optimistic': ['optimistic', 'hopeful', 'positive', 'confident'],
    'Bored': ['bored', 'uninterested', 'restless', 'dull'],
    'Apathetic': ['apathetic', 'indifferent', 'unconcerned', 'detached'],
    'Energetic': ['energetic', 'active', 'lively', 'vibrant']
}
targeted_option = {}

list_of_option = []
for option in options:   
    synonyms = emotion_keywords[option]
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    list = []
    for word in synonyms:
        tokens = tokenizer.tokenize(word)
        list.append(tokens[0])
        list_of_option.append(tokens[0])
        #print(tokens)
    targeted_option[option] = list
print(targeted_option)
def check_argument_type(arg):
    # Check if the argument is a string
    if isinstance(arg, str):
        return True
    else:
        print("The argument is neither a string nor a list.")
        print(arg)
categorize_emotion("deplorable")


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


def calculate_results_sequential(prompt, weeks, tokenizer, options):
    """Process one model at a time for all prompts"""
    epoch_num = 2
    print(prompt)
    print(options)
    # Initialize DataFrames with proper structure
    all_results = {}
    df = pd.DataFrame(index=options)
    all_results[prompt] = df
    results =  pd.DataFrame([])
    for week in tqdm(weeks, desc="Processing weeks"):
        print(f"\nProcessing week: {week}")
        model_name = "/pfs/work7/workspace/scratch/ma_mislam-newthesis/ma_mislam-Thesis_Touhidul/new1/training_batch_new/BERTweet_retrain_"+week
        model_name = getcheckpoint(model_name, epoch_num)
        pipe = pipeline(
                "fill-mask", 
                tokenizer=tokenizer, 
                model=model_name, 
                top_k=100, 
                targets=list_of_option  # ✅ Fixed incorrect syntax
            )
        output = pipe(prompt)
        print(week)
        # Initialize results dictionary if not defined
        # Initialize results dictionary if not defined
        week_results = []  # Stores scores for each emotion category
        for option in options:  # Iterate over emotion categories like "Sad", "Happy", etc.
            score = 0.0
        
            for element in output:  # Output is a list of strings with predictions
                token_str = element['token_str'] if isinstance(element, dict) else element
                token_score = element['score'] if isinstance(element, dict) else 1.0  # Default score to 1.0 if no score is found
        
                # If the predicted word falls under the current category, add its score
                if categorize_emotion(token_str) == option:
                    score += token_score
        
            print(f"{option}: {score}")  # Print category scores
            week_results.append(score)  # Store category score
        
        # Store weekly results
        results[week] = week_results
        #print(week_results)
    results.index.name = 'Emotion'
    results.columns = pd.to_datetime(
        [f"{col}_4" for col in results.columns],  # Add a day (Monday) for context
        format="%Y_%W_%w")
    return results
    
def calculate_result(prompts, weeks, tokenizer, options):
    result_list = []
    for prompt in prompts:
        results = calculate_results_sequential(prompt,weeks,tokenizer,options) 
        #here crease the average of results
        result_list.append(results)
        print(prompt)
        print(results)
    return result_list
def main():
    # Initialize
    epoch_num = 1
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    years = [2019, 2020, 2021, 2022, 2023]
    weeks = getWeek(years)
    #weeks= [ '2019_23','2019_24']
    # Create output directories if they don't exist
    os.makedirs("intermediate_results", exist_ok=True)
    os.makedirs("ten/withsim/Single", exist_ok=True)
    os.makedirs("ten/withsim/Mean", exist_ok=True)
    os.makedirs("ten/withsim/Median", exist_ok=True)
    
    print(f"Starting processing for {len(weeks)} weeks...")
    
    # Process all weeks sequentially
    result_list = calculate_result(prompts, weeks, tokenizer, options)
    
        
    # Save intermediate results
    print("Finail Reulst",result_list)
    for results in enumerate(result_list):
        print("Each Results", results)
        #results.to_csv(f"intermediate_results/prompt_{i}_raw_results.csv")
    
    # Calculate and save different averages
    results_single = get_average(result_list, 'single')
    results_single.to_csv(f"ten/withsim/Single/newpanxsingle_query_epoch{epoch_num}.csv")
    
    results_average = get_average(result_list, 'average')
    results_average.to_csv(f"ten/withsim/Mean/newpanxMulti_mean_query_epoch{epoch_num}.csv")
    results_median = get_average(result_list, 'median')
    results_median.to_csv(f"ten/withsim/Median/newpanxMulti_median_query_epoch{epoch_num}.csv")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()