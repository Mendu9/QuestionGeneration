# QuestionGeneration

This project demonstrates AI-driven workflow for summarizing lecture content, generating questions, and providing a basic grading mechanism. It uses NLP models (summarization, question generation) integrated into a Flask web application.

## Features

1. **AI-Powered Summarization**  
   Upload a lecture file (PDF, DOCX, PPTX), and the application extracts the text, preprocesses it, and summarizes it using a transformer-based summarization model (`facebook/bart-large-cnn`).

2. **Question Generation from AI Models**  
   - **6 Questions**: Questions from chunked segments of the processed text, ensuring coverage of various topics in the lecture.

3. **Interactive UI Flow**  
   - **Upload Page**: The user uploads a file. A large image (logo) is displayed at the top, with the upload button below, and a footer credit line at the bottom.
   - ![image](https://github.com/user-attachments/assets/759eccf8-e0c3-4a59-9d4a-66ec5cc442d4)
   - **Processed Page (Summary & Original Text)**: Displays the original text, summary, and extracted links. User can choose to upload another file or proceed to questions.
   - ![image](https://github.com/user-attachments/assets/5d455cea-c503-46d0-b4c8-1de2777c7c32)
   - ![image](https://github.com/user-attachments/assets/f7c77d93-e431-409d-a16b-14984f7c176b)

   - **Questions Page**: Lists the generated questions with a text area for each answer and a "Grade Answer" button.
   - ![image](https://github.com/user-attachments/assets/05a02ce7-215b-453a-82d1-bcad459d6206)


4. **Placeholder Grading Logic**  
   Currently, grading is simplistic:
   - If the student's answer shares any word with the context, it returns "Good job!"
   - Otherwise, "Try again!"
   - ![image](https://github.com/user-attachments/assets/5cefe53e-4235-459b-a092-5e21db84e7ef)
   Future enhancements can include semantic similarity-based scoring.

## Future Improvements

- **Better Question Quality**: Experiment with other question generation models or fine-tune existing ones.
- **Advanced Grading**: Implement semantic similarity for more accurate feedback.
- **Enhanced UI/UX**: Improve styling, add loading indicators, and refine the user experience.

## Installation & Running

1. **Install Dependencies**:
   ```bash
   pip install flask PyPDF2 python-docx python-pptx nltk spacy transformers keybert rake-nltk yake
