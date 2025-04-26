import React, { useState } from "react";
import {
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
} from "@mui/material";
import { Section } from "../api.ts"; // Adjust the import based on your API structure

interface QnAProps {
  documentId: string;
  selectedSection: Section | null;
}

interface QAItem {
  question: string;
  answer: string;
}

const QnA: React.FC<QnAProps> = ({ documentId, selectedSection }) => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [qaHistory, setQAHistory] = useState<QAItem[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || !selectedSection) return;

    setLoading(true);
    try {
      const answer = await askQuestion(
        documentId,
        selectedSection.title,
        question
      );
      setQAHistory((prev) => [...prev, { question, answer }]);
      setQuestion("");
    } catch (error) {
      console.error("Error asking question:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Typography variant="h6" color="primary" gutterBottom>
        Ask Questions
      </Typography>
      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this section..."
          disabled={!selectedSection || loading}
          margin="normal"
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={!selectedSection || loading || !question.trim()}
          sx={{ mt: 1 }}
        >
          {loading ? <CircularProgress size={24} /> : "Ask"}
        </Button>
      </form>
      <List sx={{ mt: 2 }}>
        {qaHistory.map((qa, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={`Q: ${qa.question}`}
              secondary={`A: ${qa.answer}`}
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default QnA;
