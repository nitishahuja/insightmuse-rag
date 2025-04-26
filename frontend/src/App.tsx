import React, { useState } from "react";
import {
  Container,
  Typography,
  Paper,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import UploadPDF from "./components/UploadPDF.tsx";
import SectionList from "./components/SectionList.tsx";
import TLDRList from "./components/TLDRList.tsx";
import QnA from "./components/QnA.tsx";
import Visualization from "./components/Visualization.tsx";
import { Section } from "./api";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#1976d2" },
    secondary: { main: "#9c27b0" },
    background: { default: "#f4f6fa" },
  },
  typography: {
    fontFamily: "Inter, Roboto, Arial, sans-serif",
  },
});

const App: React.FC = () => {
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [sections, setSections] = useState<Section[]>([]);
  const [selectedSection, setSelectedSection] = useState<Section | null>(null);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
          <Typography
            variant="h3"
            color="primary"
            fontWeight={700}
            gutterBottom
          >
            InsightMuse
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" gutterBottom>
            Research Paper Summarization, QnA, and Visualization Platform
          </Typography>
        </Paper>
        <Box mb={4}>
          <UploadPDF
            onUploadSuccess={(docId, secs) => {
              setDocumentId(docId);
              setSections(secs);
              setSelectedSection(secs[0] || null);
            }}
          />
        </Box>
        {documentId && sections.length > 0 && (
          <>
            <SectionList
              sections={sections}
              selected={selectedSection}
              onSelect={setSelectedSection}
            />
            <Box mt={4}>
              <TLDRList documentId={documentId} sections={sections} />
            </Box>
            <Box mt={4}>
              <QnA documentId={documentId} selectedSection={selectedSection} />
            </Box>
            <Box mt={4}>
              {selectedSection && (
                <Visualization
                  documentId={documentId}
                  section={selectedSection}
                />
              )}
            </Box>
          </>
        )}
      </Container>
    </ThemeProvider>
  );
};

export default App;
