import uvicorn 
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "ML Backend is running with FastAPI!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    df = pd.read_csv(file_path)

    # Generate basic stats
    stats = {
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "mean": df.mean(numeric_only=True).to_dict(),
        "median": df.median(numeric_only=True).to_dict()
    }

    return {"message": "File uploaded successfully", "stats": stats}

@app.get("/visualize")
def visualize_data():
    file_list = os.listdir(UPLOAD_FOLDER)
    if not file_list:
        return {"error": "No files uploaded"}
    
    file_path = os.path.join(UPLOAD_FOLDER, file_list[0])
    df = pd.read_csv(file_path)

    # Histogram of the first numeric column
    num_col = df.select_dtypes(include=['number']).columns[0]
    plt.figure(figsize=(8, 5))
    sns.histplot(df[num_col], bins=30, kde=True)
    plt.title(f"Distribution of {num_col}")
    plt.savefig("static/histogram.png")

    return {"message": "Visualization generated!", "histogram": "static/histogram.png"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=5000, reload=True)
