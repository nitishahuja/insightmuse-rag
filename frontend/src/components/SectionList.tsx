import React from "react";
import {
  List,
  ListItemButton,
  ListItemText,
  Paper,
  Typography,
} from "@mui/material";
import { Section } from "../api.ts";
interface SectionListProps {
  sections: Section[];
  selected: Section | null;
  onSelect: (section: Section) => void;
}

const SectionList: React.FC<SectionListProps> = ({
  sections,
  selected,
  onSelect,
}) => {
  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Typography variant="h6" color="primary" gutterBottom>
        Sections
      </Typography>
      <List>
        {sections.map((section) => (
          <ListItemButton
            key={section.index}
            selected={selected?.index === section.index}
            onClick={() => onSelect(section)}
          >
            <ListItemText primary={section.title} secondary={section.preview} />
          </ListItemButton>
        ))}
      </List>
    </Paper>
  );
};

export default SectionList;
