import React, { useEffect, useState } from "react";
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  CircularProgress,
  Box,
} from "@mui/material";
import { getTLDRs, TLDR, Section } from "../api.ts";

interface TLDRListProps {
  documentId: string;
  sections: Section[];
}

const TLDRList: React.FC<TLDRListProps> = ({ documentId, sections }) => {
  const [tldrs, setTldrs] = useState<TLDR[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    let cancelled = false;
    const pollTLDRs = async () => {
      try {
        const data = await getTLDRs(documentId);
        if (!cancelled) {
          setTldrs(data);
          setLoading(false);
          if (data.some((t) => t.status === "pending")) {
            interval = setTimeout(pollTLDRs, 2000);
          }
        }
      } catch {
        setLoading(false);
      }
    };
    pollTLDRs();
    return () => {
      cancelled = true;
      if (interval) clearTimeout(interval);
    };
  }, [documentId, sections]);

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Typography variant="h6" color="primary" gutterBottom>
        Section TLDRs
      </Typography>
      {loading && <CircularProgress />}
      <List>
        {tldrs.map((tldr, idx) => (
          <ListItem key={idx} alignItems="flex-start">
            <ListItemText
              primary={sections[idx]?.title || `Section ${idx + 1}`}
              secondary={
                tldr.status === "ready" ? (
                  <Box>
                    <Typography variant="body2" color="text.primary">
                      {tldr.tldr}
                    </Typography>
                  </Box>
                ) : tldr.status === "pending" ? (
                  <Chip label="Generating..." color="warning" size="small" />
                ) : (
                  <Chip label="Error" color="error" size="small" />
                )
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default TLDRList;
