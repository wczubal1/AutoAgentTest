import logging
import os
import sys
import autogen
#from llm_config import config_list
from autogen import ConversableAgent, UserProxyAgent
from openai import OpenAI
#from fetch_restaurant_data import fetch_restaurant_data
#from calculate_overall_score import calculate_score, calculate_overall_score
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.function_utils import get_function_schema
from typing import Annotated, Dict, List
from math import sqrt
import numpy as np
from pathlib import Path

config_list=[{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

def fetch_restaurant_data(restaurant_name: Annotated[str, "Restautan name to find review for."]):
    """
    Searches for lines in the given text file that start with the specified restaurant name
    followed by a dot (case-insensitive). Returns a list of dictionaries with the restaurant
    name as the key (without the dot) and the rest of the line as the value.

    Args:
        restaurant_name (str): The restaurant name to search for.

    Returns:
        list: A list of dictionaries with the restaurant name as the key and the rest of the line as the value.
              Returns an empty list if the restaurant is not found.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the other file
    file_path = os.path.join(current_dir, "restaurant-data.txt")  
    results = []
    search_prefix = f"{restaurant_name}.".lower()  # Lowercase for case-insensitive search

    try:
        with open(file_path,'r') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.lower().startswith(search_prefix):
                    # Extract the portion after the restaurant name and dot
                    rest_of_line = stripped_line[len(search_prefix):].lstrip()
                    # Preserve the original restaurant name's casing
                    actual_restaurant_name = stripped_line[:len(search_prefix) - 1]
                    results.append({actual_restaurant_name: rest_of_line})
        return results

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def calculate_score(review_text: Annotated[str, "review of a restaurant to calculate score."]):
    """
    Extracts food_score and customer_service_score from a restaurant review.

    Args:
        review_text (str): The text of the restaurant review.

    Returns:
        dict: A dictionary containing 'food_score' and 'customer_service_score'.
              Example: {'food_score': 3, 'customer_service_score': 2}
    """
    # Normalize review text to lowercase for uniformity
    review_text = review_text.lower()

    # Define keyword to score mapping
    keywords_to_scores = {
        'awful': 1,
        'horrible': 1,
        'disgusting': 1,
        'bad': 2,
        'unpleasant': 2,
        'offensive': 2,
        'average': 3,
        'uninspiring': 3,
        'forgettable': 3,
        'good': 4,
        'enjoyable': 4,
        'satisfying': 4,
        'awesome': 5,
        'incredible': 5,
        'amazing': 5
    }

    # Aspects and possible indicative words
    food_related = ['food', 'menu', 'dish', 'meal', 'fare', 'options','donuts','chicken','burgers']
    customer_service_related = ['customer service', 'service', 'staff', 'waiter', 'waitress', 'server', 'team']

    # Initialize scores
    scores = {'food_score': None, 'customer_service_score': None}

    # Process text by splitting into sentences to properly associate adjectives with aspects
    sentences = review_text.split('. ')

    for sentence in sentences:
        for keyword, score in keywords_to_scores.items():
            if keyword in sentence:
                # Check for food context keywords
                if any(food_context in sentence for food_context in food_related):
                    scores['food_score'] = score
                # Check for customer service context keywords
                elif any(service_context in sentence for service_context in customer_service_related):
                    scores['customer_service_score'] = score

        # If both scores are found, break early
        if all(v is not None for v in scores.values()):
            break

    # Ensure both scores were found
    if None in scores.values():
        raise ValueError("Could not extract scores for both 'food' and 'customer service' aspects. Check if both are described using defined keywords.")

    return scores

def square(list):
    return [i ** 2 for i in list]


def calculate_overall_score(restaurant_name: str, food_scores: list[int], customer_service_scores: list[int]) -> dict[str, float]:
    N=len(food_scores)
    overall_score=sum(np.sqrt( np.multiply(square(food_scores),customer_service_scores)) * 1/(N * sqrt(125)) * 10)
    return  {restaurant_name: overall_score}


def main(user_query: str):

    llm_config = {
        "config_list": config_list,
    }

    # the main entrypoint/supervisor agent
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "last_n_messages": 1,
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    ##entrypoint_agent = ConversableAgent("entrypoint_agent", 
    ##                                    system_message="You fetch review data for a restaurant and send it to review_agent", 
    ##                                    llm_config=llm_config)
    ##entrypoint_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    ##UserProxyAgent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    # https://microsoft.github.io/autogen/0.2/docs/notebooks/gpt_assistant_agent_function_call

    # Assistant API Tool Schema for get_dad_jokes
    fetch_restaurant_data_schema = get_function_schema(
        fetch_restaurant_data,
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant.",
    )

    calculate_score_schema = get_function_schema(
        calculate_score,
        name="calculate_score",
        description="Calculates score for a specific restaurant.",
    )

    calculate_overall_score_schema = get_function_schema(
        calculate_overall_score,
        name="calculate_overall_score",
        description="Calculates overall score for a specific restaurant.",
    )

    entrypoint_agent = GPTAssistantAgent(
        name="entrypoint_agent",
        instructions="""
        As 'entrypoint_agent', your primary role is to fetch all reviews data for a restaurant and review_agent will score it, your task is as follows:

        1. Use the 'fetch_restaurant_data' function to search for each restaurant review for the provided restaurant name.
        2. Remember that restaurant name can be spelled slightly differently for examle "In N Out" as "In-N-Out". 
        3. Present all these reviews to review_agent. 
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [fetch_restaurant_data_schema],
        },
    )

    entrypoint_agent.register_function(
        function_map={
            "fetch_restaurant_data": fetch_restaurant_data,
        },
    )

    review_agent = GPTAssistantAgent(
        name="review_agent",
        instructions="""
        As 'review_agent', You read each review of a restaurant and score each one using 'calculate_score' function. your task is as follows:

        1. Use the 'calculate_score' to provide score for the restaurant based on the review.
        2. You score separately each review for the same restaurant.
        3. Once you score the reviews you communicate the results to built_list agent
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [calculate_score_schema],
        },
    )

    review_agent.register_function(
        function_map={
            "calculate_score": calculate_score,
        },
    )

    built_list = GPTAssistantAgent(
        name="built_list",
        instructions="""
        As 'built_list' agent you read each review of a restaurant and build two python lists "food_scores" and "customer_service_scores". Your task is as follows:

        1. You check if both food_score and  customer service score have numerical values
        2. If any of the scores is missing or error you move to the next review
        3. If both scores are available you add food score to "food_scores" list and customer service score to "customer_service_scores" list.
        4. Once you go through all reviews you send both lists to 'score_agent'.
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
        },
    )

    score_agent = GPTAssistantAgent(
        name="score_agent",
        instructions="""
        As 'score_agent', You take python list "food_scores" and python list "customer_service_scores" from 'built_list' agent and calculate overall score. your task is as follows:

        1. Use the function 'calculate_overall_score' to provide final score for the restaurant based on "food_scores" and "customer_service_scores" that you get from review_agent
        2. Call function 'calculate_overall_score'. Arguments are restaurant_name as str, food_scores as list, customer_service_scores as list 
	3. Share the result up to 3 decimals with the user
        3. Reply "TERMINATE" in the end when you finished scoring the restaurant.
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [calculate_overall_score_schema],
        },
    )

    score_agent.register_function(
        function_map={
            "calculate_overall_score": calculate_overall_score,
        },
    )

    groupchat = autogen.GroupChat(agents=[user, entrypoint_agent, review_agent, built_list, score_agent], messages=[], max_round=15)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

    result = user.initiate_chat(group_chat_manager, message=user_query,summary_method="last_msg")

# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
    #main("Krispy Kreme")
