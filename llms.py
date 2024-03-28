import os
from openai import OpenAI

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
      stop=stop
    )
    
    return response.choices[0].text