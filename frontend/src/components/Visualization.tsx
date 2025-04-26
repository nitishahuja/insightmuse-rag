import React, { useState } from "react";
import {
  Paper,
  Typography,
  Box,
  Button,
  CircularProgress,
} from "@mui/material";
import { Section, getVisualizationUrl } from "../api.ts";

interface VisualizationProps {
  documentId: string;
  section: Section;
}

const Visualization: React.FC<VisualizationProps> = ({
  documentId,
  section,
}) => {
  const [loading, setLoading] = useState(false);
  const [imgKey, setImgKey] = useState(0); // For force reload

  const handleRegenerate = () => {
    setLoading(true);
    setImgKey((k) => k + 1);
    setTimeout(() => setLoading(false), 1000); // Simulate loading
  };

  const imageUrl =
    getVisualizationUrl(documentId, section.title) + `&key=${imgKey}`;

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Typography variant="h6" color="primary" gutterBottom>
          Visualization: {section.title}
        </Typography>
        <Button
          variant="outlined"
          size="small"
          onClick={handleRegenerate}
          disabled={loading}
        >
          Regenerate
        </Button>
      </Box>
      <Box mt={2} textAlign="center">
        {loading ? (
          <CircularProgress />
        ) : (
          <img
            src={imageUrl}
            alt={`Visualization for ${section.title}`}
            style={{
              maxWidth: "100%",
              maxHeight: 400,
              borderRadius: 8,
              border: "1px solid #eee",
            }}
            onLoad={() => setLoading(false)}
            onError={() => setLoading(false)}
          />
        )}
      </Box>
    </Paper>
  );
};

export default Visualization;
