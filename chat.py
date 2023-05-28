import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')


class Chat:
    def __init__(self, model='gpt-3.5-turbo'):
        self.model = model

    def chat(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        print('Waiting for response from chatGPT ...')
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message['content']


def main():
    text = f"""
        """
    prompt = f"""
    Identify the emotion of each line in the software engineering text delimited by triple backticks.
    ```
    {text}
    ```
    Classify each line of the text as '1' or '-1' or '0':
    - '1' means it expresses positive emotion
    - '-1' means it expresses negative emotion
    - '0' means it is neutral
    Provide them in JSON format with the following keys:
    index, message, label, reason.
    Remember that:
    - 'index' starts from 0.
    - 'label' is the classification result.
    - 'message' is the first 3 words of the corresponding line in the text delimited by triple backticks.
    - 'reason' gives a brief summary and explains how it is classified based on the summary within 10 words.
    """
    chat = Chat()
    response = chat.chat(prompt=prompt)
    print(response)


if __name__ == '__main__':
    main()
