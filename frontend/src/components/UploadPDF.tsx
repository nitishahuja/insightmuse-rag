import React, { useRef, useState } from "react";
import { Box, Button, Typography, LinearProgress, Alert } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { uploadPDF, Section } from "../api.ts";

interface UploadPDFProps {
  onUploadSuccess: (documentId: string, sections: Section[]) => void;
}

const UploadPDF: React.FC<UploadPDFProps> = ({ onUploadSuccess }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadPDF(file);
      onUploadSuccess(res.document_id, res.sections);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Failed to upload PDF.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box textAlign="center">
      <input
        type="file"
        accept="application/pdf"
        style={{ display: "none" }}
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <Button
        variant="contained"
        color="primary"
        startIcon={<CloudUploadIcon />}
        onClick={() => fileInputRef.current?.click()}
        disabled={uploading}
        sx={{ minWidth: 200 }}
      >
        Upload PDF
      </Button>
      {uploading && <LinearProgress sx={{ mt: 2 }} />}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        Only PDF files are supported. TLDRs will be generated automatically.
      </Typography>
    </Box>
  );
};

export default UploadPDF;
