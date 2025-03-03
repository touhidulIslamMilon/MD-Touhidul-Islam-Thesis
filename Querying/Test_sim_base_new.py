"""
Author: MD Touhidul Islam
Description: BERT model querying script for emotion analysis and temporal mood tracking

This script evaluates fine-tuned BERT models on various prompts related to emotional
states. It processes multiple prompts across different time periods and aggregates
the results to analyze temporal trends in emotional responses.
"""

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

def getWeek(years):
    """
    Generate a list of week identifiers for specified years.
    
    The function handles different ranges for different years:
    - 2019: Weeks 22-53 only
    - 2023: Weeks 1-9 only
    - Other years: Full weeks 1-52
    
    Args:
        years: List of years to generate week identifiers for
        
    Returns:
        weeks: List of formatted week identifiers in 'YYYY_WW' format
    """
    weeks = []
    for year in years:
        if year == 2019:
            # For 2019, only include weeks 22-53
            for i in range(22, 53):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
        elif year == 2023:
            # For 2023, only include weeks 1-9
            for i in range(1, 10):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
        else:
            # For other years, include all weeks
            for i in range(1, 53):
                Week = f"{year}_{i:02d}"
                weeks.append(Week)
    print(f"Generated {len(weeks)} weeks")
    return weeks
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
options = ["Sad", "Happy", "Content", "Other", "Stressed", "Lonely", "Inspired", "Frustrated", "Optimistic", "Bored", "Apathetic", "Energetic"]

targeted_option = {}
list_of_option = []
for option in options:   
    synonyms = []
    for syn in wordnet.synsets(option):
        for i in syn.lemmas():
            synonyms.append(i.name())
    print(set(synonyms))
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
    """
    Validate that the argument is a string.
    
    Args:
        arg: The argument to check
        
    Returns:
        bool: True if argument is a string, False otherwise
    """
    # Check if the argument is a string
    if isinstance(arg, str):
        return True
    else:
        print("The argument is neither a string nor a list.")
        print(arg)
        return False
        
def categorize_emotion(arg):
    """
    Categorize a term into an emotion category based on predefined keywords.
    
    Args:
        arg: String to categorize
        
    Returns:
        str: The matched emotion category or 'Unknown' if no match found
    """
    check_argument_type(arg)
    for emotion, keywords in targeted_option.items():
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

def getcheckpoint(checkpoint_dir, epoch_num): 
    """
    Find and return the appropriate checkpoint directory based on epoch number.
    
    Args:
        checkpoint_dir: Base directory containing model checkpoints
        epoch_num: Epoch number to determine which checkpoint to use
        
    Returns:
        model_dir: Full path to the selected checkpoint directory
    """
    # Step 1: List all directories (checkpoints) in the directory
    available_checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    # Step 2: Sort the checkpoints by the epoch number (assuming they follow the pattern 'checkpoint-epoch-X')
    # If your checkpoints follow a pattern like 'checkpoint-epoch-1', 'checkpoint-epoch-2', etc.
    available_checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sorting by the last part after 'epoch-'
    
    # Step 3: Select the appropriate checkpoint based on epoch_num parameter
    if epoch_num == 1:
        checkpoint = available_checkpoints[-1]  # Last checkpoint
    elif epoch_num == 2:
        checkpoint = available_checkpoints[1]   # Second checkpoint
    else:
        checkpoint = available_checkpoints[2]   # Third checkpoint
        
    model_dir = checkpoint_dir+'/'+checkpoint
    print(f"Loading the latest checkpoint: {model_dir}")
    return model_dir
def get_average(dfs, selection_type):
    """
    Calculate aggregate statistics from multiple dataframes of results.
    
    This function can compute the mean, median, or simply return a single dataframe
    based on the selection_type parameter. It combines multiple result dataframes
    from different prompts or time periods into a single summary dataframe.
    
    Args:
        dfs: List of pandas DataFrames containing results to aggregate
        selection_type: Type of aggregation - 'average', 'median', or 'single'
        
    Returns:
        DataFrame: Aggregated results based on specified selection_type
    """
    try:
        # Print shape and sample of input dataframes
        print(f"\nNumber of dataframes: {len(dfs)}")
        for i, df in enumerate(dfs):
            print(f"\nDataFrame {i+1} shape: {df.shape}")
            print("Sample of values:")
            print(df.iloc[:5, :2])  # Show first 5 rows, 2 columns
        
        # Combine dataframes based on selection type
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
            result = dfs[0]  # Just use the first dataframe as is
        
        # Print intermediate results for debugging
        print("\nResult shape:", result.shape)
        print("Sample of results:")
        print(result.iloc[:5, :2])
        
        # Ensure proper index naming
        result.index.name = 'Emotion'
        
        # Map special tokens back to original names
        # (This appears to be a placeholder for future implementation)
        
        return result
        
    except Exception as e:
        print(f"Error in get_average: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Define prompts and options


def calculate_results_sequential(prompt, weeks, tokenizer, options):
    """
    Process one model at a time for all prompts and weeks, calculating emotion scores.
    
    This function loads fine-tuned models for each week, runs the specified prompt through
    the masked language model pipeline, and calculates scores for different emotion categories
    based on the model's predictions.
    
    Args:
        prompt: The prompt text with <mask> token to query the model
        weeks: List of week identifiers to process
        tokenizer: Tokenizer for the model
        options: List of emotion categories to score
        
    Returns:
        DataFrame: Results with emotion categories as rows and weeks as columns
    """
    epoch_num = 2  # Which model checkpoint epoch to use
    print(prompt)
    print(options)
    
    # Initialize DataFrames with proper structure
    all_results = {}
    df = pd.DataFrame(index=options)
    all_results[prompt] = df
    results = pd.DataFrame([])
    
    # Process each week sequentially
    for week in tqdm(weeks, desc="Processing weeks"):
        print(f"\nProcessing week: {week}")
        
        # Construct path to model checkpoint for this week
        model_name = "/pfs/work7/workspace/scratch/ma_mislam-newthesis/ma_mislam-Thesis_Touhidul/new1/training_batch_new/BERTweet_retrain_"+week
        model_name = getcheckpoint(model_name, epoch_num)
        
        # Set up the fill-mask pipeline with the week's model
        pipe = pipeline(
                "fill-mask", 
                tokenizer=tokenizer, 
                model=model_name, 
                top_k=100,  # Get top 100 predictions
                targets=list_of_option  # Target specific tokens
            )
        
        # Run the prompt through the model
        output = pipe(prompt)
        print(week)
        
        # Calculate scores for each emotion category
        week_results = []  # Stores scores for each emotion category
        for option in options:  # Iterate over emotion categories like "Sad", "Happy", etc.
            score = 0.0
        
            # Process each predicted token
            for element in output:  # Output is a list of dictionaries with predictions
                token_str = element['token_str'] if isinstance(element, dict) else element
                token_score = element['score'] if isinstance(element, dict) else 1.0  # Default score if not found
        
                # If the predicted word falls under the current emotion category, add its score
                if categorize_emotion(token_str) == option:
                    score += token_score
        
            print(f"{option}: {score}")  # Print category scores for debugging
            week_results.append(score)  # Store category score
        
        # Store weekly results as a column in the results DataFrame
        results[week] = week_results
    
    # Set index name and convert columns to datetime format for time series analysis
    results.index = options
    results.index.name = 'Emotion'
    results.columns = pd.to_datetime(
        [f"{col}_4" for col in results.columns],  # Add a day (Monday) for context
        format="%Y_%W_%w")
    
    return results
    
def calculate_result(prompts, weeks, tokenizer, options):
    """
    Process multiple prompts across all weeks and collect results.
    
    This function iterates through each provided prompt, calls the
    calculate_results_sequential function, and collects all results
    into a list for further processing.
    
    Args:
        prompts: List of prompts to evaluate
        weeks: List of week identifiers to process
        tokenizer: The tokenizer for the model
        options: List of emotion categories to score
        
    Returns:
        list: List of DataFrames containing results for each prompt
    """
    result_list = []
    for prompt in prompts:
        # Process each prompt across all weeks
        results = calculate_results_sequential(prompt, weeks, tokenizer, options) 
        # Add results to the list
        result_list.append(results)
        print(prompt)
        print(results)
    return result_list

def main():
    """
    Main function to orchestrate the entire process of querying models and collecting results.
    
    This function initializes the necessary components, processes all prompts across
    all time periods, aggregates results, and saves them to specified output files.
    The function handles:
    1. Initialization of tokenizer and time periods
    2. Creation of output directories
    3. Processing prompts across weeks
    4. Saving raw and aggregated results
    """
    # Initialize models and time periods
    epoch_num = 1
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    years = [2019, 2020, 2021, 2022, 2023]
    weeks = getWeek(years)
    #weeks= [ '2019_23','2019_24']  # Uncomment for testing with fewer weeks
    
    # Create output directories if they don't exist
    os.makedirs("intermediate_results", exist_ok=True)
    os.makedirs("ten/withsim/Single", exist_ok=True)
    os.makedirs("ten/withsim/Mean", exist_ok=True)
    os.makedirs("ten/withsim/Median", exist_ok=True)
    
    print(f"Starting processing for {len(weeks)} weeks...")
    
    # Process all prompts across all weeks
    result_list = calculate_result(prompts, weeks, tokenizer, options)
    
    # Print and save intermediate results
    print("Final Result", result_list)
    for results in enumerate(result_list):
        print("Each Results", results)
        #results.to_csv(f"intermediate_results/prompt_{i}_raw_results.csv")
    
    # Calculate and save different statistical aggregations
    # 1. Single results (first prompt only)
    results_single = get_average(result_list, 'single')
    results_single.to_csv(f"ten/withsim/Single/newsimsingle_query_epoch{epoch_num}.csv")
    
    # 2. Mean of all prompts
    results_average = get_average(result_list, 'average')
    results_average.to_csv(f"ten/withsim/Mean/newsimMulti_mean_query_epoch{epoch_num}.csv")
    
    # 3. Median of all prompts
    results_median = get_average(result_list, 'median')
    results_median.to_csv(f"ten/withsim/Median/newsimMulti_median_query_epoch{epoch_num}.csv")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()