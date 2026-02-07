from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client (will be created per request to avoid initialization issues)
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return OpenAI(api_key=api_key)

# Request model
class DiagnosticTestRequest(BaseModel):
    course_name: str = Field(..., description="Name of the course")
    subject: str = Field(..., description="Subject area")
    target_grade_level: str = Field(..., description="Target grade or level")
    number_of_mcq: int = Field(..., ge=1, le=50, description="Number of MCQ questions (1-50)")

# Response models
class MCQOption(BaseModel):
    option: str
    text: str

class MCQQuestion(BaseModel):
    question_number: int
    question: str
    options: List[MCQOption]
    correct_answer: str
    explanation: str

class DiagnosticTestResponse(BaseModel):
    course_name: str
    subject: str
    target_grade_level: str
    total_questions: int
    questions: List[MCQQuestion]

@app.get("/")
async def root():
    return {
        "message": "Diagnostic Test Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/generate-test": "POST - Generate a diagnostic test",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the API is running and OpenAI key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "openai_configured": bool(api_key)
    }

@app.post("/generate-test", response_model=DiagnosticTestResponse)
async def generate_diagnostic_test(request: DiagnosticTestRequest):
    """
    Generate a diagnostic test using OpenAI GPT
    """
    try:
        # Initialize OpenAI client
        client = get_openai_client()
        
        # Create the prompt for ChatGPT
        prompt = f"""
You are an expert educational assessment creator. Create a diagnostic test with the following specifications:

Course Name: {request.course_name}
Subject: {request.subject}
Target Grade/Level: {request.target_grade_level}
Number of MCQ Questions: {request.number_of_mcq}

Generate {request.number_of_mcq} multiple-choice questions that assess prerequisite knowledge for this course. 
Each question should:
1. Test fundamental concepts relevant to the course
2. Have 4 options (A, B, C, D)
3. Have only one correct answer
4. Include a brief explanation of the correct answer

Return the response in the following JSON format:
{{
  "questions": [
    {{
      "question_number": 1,
      "question": "Question text here",
      "options": [
        {{"option": "A", "text": "Option A text"}},
        {{"option": "B", "text": "Option B text"}},
        {{"option": "C", "text": "Option C text"}},
        {{"option": "D", "text": "Option D text"}}
      ],
      "correct_answer": "A",
      "explanation": "Explanation of why this is correct"
    }}
  ]
}}

Ensure the JSON is valid and properly formatted.
"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to "gpt-4o" or "gpt-4" for better results
            messages=[
                {"role": "system", "content": "You are an expert educational assessment creator. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        # Extract the response content
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Parse the JSON response
        test_data = json.loads(content)

        # Construct the response
        diagnostic_test = DiagnosticTestResponse(
            course_name=request.course_name,
            subject=request.subject,
            target_grade_level=request.target_grade_level,
            total_questions=request.number_of_mcq,
            questions=test_data["questions"]
        )

        return diagnostic_test

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse GPT response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)