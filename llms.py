import os
from openai import OpenAI
import anthropic


def GPT3(prompt, stop=["\n"]):
    # legacy: model="text-davinci-002"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.completions.create(
      model="gpt-3.5-turbo-instruct",
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop,
    )
    
    return response.choices[0].text


def GPT4(prompt, stop=["\n"]):
    # legacy: model="text-davinci-002"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.chat.completions.create(
      model="gpt-4-0125-preview",
      messages=[
            {
                "role": "system",
                "content": "Complete the prompt."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop,
    )
    
    return response.choices[0].message.content


def Claude3Opus(prompt, stop=["\n"]):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],)

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        temperature=0,
        stop_sequences=stop,
        system="Complete the prompt.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
    )
    return message.content[0].text