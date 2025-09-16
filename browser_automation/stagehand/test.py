import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from stagehand import StagehandConfig, Stagehand

# Load environment variables
load_dotenv()

# Define Pydantic models for structured data extraction
class Company(BaseModel):
    name: str = Field(..., description="Company name")
    description: str = Field(..., description="Brief company description")

class Companies(BaseModel):
    companies: list[Company] = Field(..., description="List of companies")
    
async def main():
    # Create configuration
    config = StagehandConfig(
        env = "LOCAL", # or BROWSERBASE
        local_browser_launch_options={
        "headless": False,
        "devtools": True
        },
        api_key=os.getenv("BROWSERBASE_API_KEY"),
        project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
        model_name="openai/gpt-5-mini-2025-08-07",
        model_api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    stagehand = Stagehand(config)
    
    try:
        print("\nInitializing 🤘 Stagehand...")
        # Initialize Stagehand
        await stagehand.init()

        if stagehand.env == "BROWSERBASE":    
            print(f"🌐 View your live browser: https://www.browserbase.com/sessions/{stagehand.session_id}")

        page = stagehand.page

        await page.goto("https://www.aigrant.com")
        
        # Extract companies using structured schema        
        companies_data = await page.extract(
          "Extract names and descriptions of 5 companies in batch 3",
          schema=Companies
        )
        
        # Display results
        print("\nExtracted Companies:")
        for idx, company in enumerate(companies_data.companies, 1):
            print(f"{idx}. {company.name}: {company.description}")

        observe = await page.observe("the link to the company Browserbase")
        print("\nObserve result:", observe)
        act = await page.act("click the link to the company Browserbase")
        print("\nAct result:", act)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        # Close the client
        print("\nClosing 🤘 Stagehand...")
        await stagehand.close()

if __name__ == "__main__":
    asyncio.run(main())