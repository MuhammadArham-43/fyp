import os
import openai
# openai.organization = "YOUR_ORG_ID"
openai.api_key = "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"

# Use a model from OpenAI (assuming "text-embedding-ada-002" exists for this example)
model_name="gpt-3.5-turbo-0125"


def chat_with_openai(prompt):
    """
    Sends the prompt to OpenAI API using the chat interface and gets the model's response.
    """
    message = {
        'role': 'user',
        'content': prompt
    }

    response = openai.ChatCompletion.create(
        model=model_name,
        temperature=0.15,
        messages=[message]
    )

    # Extract the chatbot's message from the response.
    # Assuming there's at least one response and taking the last one as the chatbot's reply.
    # print("Choices", c)
    chatbot_response = response.choices[0].message['content']
    with open("chatbot_response.txt", "a") as f:
        f.write(chatbot_response)
        print("Data Writing Done")
    # return chatbot_response.strip()


def main():
    """
    Main interaction loop for the chatbot.
    """
    # print("Welcome to Chatbot! Type 'quit' to exit.")

    # user_input = ""
    # while user_input.lower() != "quit":
        # user_input = input("You: ")

        # if user_input.lower() != "quit":
    prompt = """
    
    I want to generate data and then use it for training a model. Dataset comprises of prompts and their corresponding optimized prompts.
    Prompt is day to day user input to the large language model, while optimized prompt focuses on the essential aspects of the prompt taileored to help large language model to get the optimum result/response from Large Language Model. 
    We can also say that otpimized prompts convey exactly the same meaning as the original prompt but in a more effective manner so one should get the best response from the large language model.
    
    You should return json format only:
    
    format = 
    {
        "prompt": "Summarize the chapter 'The French Revolution' from the history textbook.",
        "optimized_prompt": "Produce a succinct summary of the chapter 'The French Revolution' from the history textbook, capturing essential events and themes."
    }
    
    GENERATE DATASET OF 300 SUCH EXAMPLES AND DONOT REPEAT, tasks should cover prompts for given subdomains:
    - querying debugging issues in computer science, specific to coding
    - academic summarization, email writing, academic writing
    - transfer one text style to other, like converting a poem to a story or vice versa
    
    OUTPUT IN JSON FORMAT ONLY NOTHING ELSE
    json:"""
    
    response = chat_with_openai(prompt)  # Pass user_input as an argument
    print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()