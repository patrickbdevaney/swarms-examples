import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from swarms.prompts.logistics import (
    Efficiency_Agent_Prompt,
    Health_Security_Agent_Prompt,
    Productivity_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
)
from swarms.structs import Agent

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: API key not found in environment variables")
    exit(1)

print("API Key Loaded:", api_key is not None)

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

class VisionAPI:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def run(self, prompt: str, image_url: str) -> Optional[str]:
        print(f"Processing: {prompt[:50]}... for image URL")
        try:
            # Combine the prompt and image URL into a single string
            full_prompt = f"{prompt}\n\nImage URL: {image_url}"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,  # Send the full prompt as a single string
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

# Initialize vision API
vision_api = VisionAPI(client=client)

# Initialize agents
agents = {
    "health_security_analysis": Agent(
        llm=vision_api,
        sop=Health_Security_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "quality_control_analysis": Agent(
        llm=vision_api,
        sop=Quality_Control_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "productivity_analysis": Agent(
        llm=vision_api,
        sop=Productivity_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "safety_analysis": Agent(
        llm=vision_api,
        sop=Safety_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "security_analysis": Agent(
        llm=vision_api,
        sop=Security_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "sustainability_analysis": Agent(
        llm=vision_api,
        sop=Sustainability_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
    "efficiency_analysis": Agent(
        llm=vision_api,
        sop=Efficiency_Agent_Prompt,
        max_loops=1,
        multi_modal=True,
    ),
}

# URL of the image for analysis
image_url = "https://raw.githubusercontent.com/patrickbdevaney/swarms-examples/main/examples/demos/logistics/factory_image1.jpg"  
# Run agents and collect results
results = {}
for key, agent in agents.items():
    print(f"\nRunning {key.replace('_', ' ').title()}...")
    try:
        task = f"Analyze the following image for {key.replace('_', ' ')}"
        result = agent.run(task, image_url)  # Pass the image URL to the agent's run method
        print(f"✓ {key.replace('_', ' ').title()} completed")
        results[key] = result
    except Exception as e:
        print(f"✗ Error in {key}: {str(e)}")
        results[key] = f"Error: {str(e)}"

# Save results to a JSON file
output_file = "factory_analysis_results.json"
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\nResults saved to {output_file}")

# Print summary
print("\nAnalysis Summary:")
for key, result in results.items():
    status = "✓ Success" if "Error" not in str(result) else "✗ Failed"
    print(f"{status}: {key.replace('_', ' ').title()}")
