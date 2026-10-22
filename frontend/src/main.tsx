import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import ProjectsPage from "./pages/ProjectsPage";
import ProjectPage from "./pages/ProjectPage";
import AnalysisPage from "./pages/AnalysisPage";
import ExportPage from "./pages/ExportPage";
import SurvivalPage from "./pages/SurvivalPage";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/projects/:id/:tab?" element={<ProjectPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/survival" element={<SurvivalPage />} />
          <Route path="/export" element={<ExportPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
