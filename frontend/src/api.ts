import axios from "axios";

const API_BASE = "http://localhost:8000"; // Change if backend runs elsewhere

export interface Section {
  title: string;
  index: number;
  preview: string;
}

export interface TLDR {
  title: string;
  status: "pending" | "ready" | "error";
  tldr: string | null;
}

export interface QnAResponse {
  answer: string;
}

export const uploadPDF = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await axios.post(`${API_BASE}/upload_pdf`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data as { document_id: string; sections: Section[] };
};

export const getSections = async (document_id: string) => {
  const res = await axios.get(`${API_BASE}/sections`, {
    params: { document_id },
  });
  return res.data.sections as Section[];
};

export const getTLDRs = async (document_id: string) => {
  const res = await axios.get(`${API_BASE}/tldr`, { params: { document_id } });
  return res.data.tldrs as TLDR[];
};

export const askQnA = async (document_id: string, question: string) => {
  const formData = new FormData();
  formData.append("document_id", document_id);
  formData.append("question", question);
  const res = await axios.post(`${API_BASE}/qna`, formData);
  return res.data as QnAResponse;
};

export const getVisualizationUrl = (
  document_id: string,
  section_title: string
) => {
  // Returns the URL to fetch the visualization image
  return `${API_BASE}/visualize?document_id=${encodeURIComponent(
    document_id
  )}&section_title=${encodeURIComponent(section_title)}`;
};
