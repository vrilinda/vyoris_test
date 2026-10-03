# Venkata Raghunadh Illinda

Hyderabad, India | +91 99121 04588 | [hi.villinda3@gmail.com](mailto:hi.villinda3@gmail.com) | [linkedin.com/in/askillinda](https://www.linkedin.com/in/askillinda/) | [github.com/vrilinda](https://github.com/vrilinda)

## Professional Summary

MLOps and AIOps engineer who applies 18 years of running mission-critical investment banking platforms to the reliability of ML and AI systems. Hands-on in Python, Bash and SQL; builds ML services (FastAPI, Flask, Streamlit), ML pipelines and MCP-based LLM agent tooling, packaged with Docker and deployed to Azure through CI/CD. Brings production-grade monitoring and incident automation (AppDynamics, Splunk, Grafana, Amelia AIOps, SLOs/SLIs, Kubernetes/AKS) to model monitoring and drift detection. MSc in AI & ML (IIIT Bangalore and Liverpool John Moores University), expected 2027.

## Core Skills

- **MLOps:** ML pipelines, model packaging and serving, model evaluation, dataset/leakage and drift-scenario testing, model maintenance via pipelines
- **AIOps / Observability:** AppDynamics, Splunk, Grafana, Amelia (AIOps platform), alerting, telemetry, SLO/SLI design, incident automation
- **Languages:** Python, Bash, SQL (PostgreSQL, Oracle PL/SQL, MSSQL, Sybase); basic Java/C# debugging
- **ML / AI:** EDA, data engineering, PyTorch, PyTorch Forecasting (Temporal Fusion Transformer), HuggingFace Transformers (FinBERT), LangChain/LangGraph, LLM prompt engineering
- **Serving / Web:** FastAPI, Flask, Streamlit, Django, HTMX
- **Agentic / MCP:** MCP server design and tool exposure, LLM tool-calling agents
- **Cloud / DevOps:** Azure (AKS, App Service, storage), Docker, Kubernetes, GitHub Actions, GitLab CI/CD, Snowflake, ETL and batch/streaming pipelines
- **Practices:** ITIL, incident and problem management, disaster-recovery planning, dashboards and operational reporting

## Professional Experience

### Associate Director | UBS Business Solutions, Hyderabad
May 2025 - Present (UBS Business Solutions LLC, Mar 2023 - Apr 2025)

- Led a team of 6 engineers supporting 40+ securities lending (stock borrow/loan) applications in investment banking
- Automated incident-management workflows for 22 applications, cutting mean time to resolve from 30 to 10 minutes
- Reduced alert noise by 35% by tuning thresholds and consolidating 250 alerts across AppDynamics and Splunk
- Defined SLOs and SLIs, with availability reporting, for 3 business-critical services rebuilt on new platforms; applied SRE practices across cloud and on-premises applications
- Built 5 Grafana (LGTM stack) dashboards used daily by application and business stakeholders; leading the migration of monitoring to Grafana and Amelia (AIOps)
- Operated AKS workloads for 2 applications rebuilt on the new platform
- Maintained CI/CD pipelines for bi-weekly sprint releases, reducing failed deployments by 5%
- Architected and supported data platforms on Azure storage and Snowflake: ETL pipelines, batch jobs and streaming services
- Mentored engineers on SRE principles, observability and cloud application strategy; led vendor evaluations

### Authorized Officer | UBS Business Solutions LLC | Nashville, TN, USA
Oct 2018 - Mar 2023

- Monitored system health and performance of trading and prime brokerage platforms, resolving issues before business impact
- Wrote and maintained automation scripts for routine operational tasks, reducing manual effort and human error
- Implemented incident-management processes that shortened response time to critical outages
- Built real-time dashboards and reporting for trading staff and management
- Contributed to disaster-recovery planning and testing for business continuity
- Acted as primary contact for major upgrades, coordinating development, QA and operations
- Audited system access and security controls with the cybersecurity team

### Earlier Roles

- **Software Development Advisor**, NTT Data (formerly Dell Services), Pune, India, May 2016 - Sep 2018
- **Senior Software Engineer**, NTT Data (formerly Dell Services), Singapore, May 2014 - May 2016
- **Senior Application Developer**, APAR Technologies, Singapore, Sep 2013 - Mar 2014
- **Senior Software Analyst**, Dell Perot Systems and Cognizant, Singapore, Jun 2009 - Sep 2013
- **Software Engineer**, CES India Pvt. Ltd., Bangalore, India, Sep 2008 - Jun 2009

## Selected AI/ML Projects

### VYORIS: Quantitative Market Analysis Platform (MSc project)
Python, FastAPI, MCP, LangChain/LangGraph, Anthropic Claude, PyTorch Forecasting (TFT), HuggingFace FinBERT, Supabase/PostgreSQL, HTMX, Azure App Service, GitHub Actions

- Built an agentic web platform for NSE/BSE stocks: a FastAPI service with a LangGraph ReAct agent (Claude) that calls four tools (market data, news sentiment, forecast, model metrics) and returns a two-audience briefing (retail and quant)
- Built an MCP server that exposes the same four tools to any MCP client over the Model Context Protocol; the web agent reuses the tool functions in-process
- Wrote the agent system prompt with a structured prompt framework (context, objective, ordered steps, audience, strict output format)
- Implemented FinBERT news-sentiment scoring (confidence-weighted, -1 to +1) and data ingestion with yfinance, Z-score outlier clipping and scaling
- Wrote TFT and LSTM-baseline training and evaluation code (RMSE, MAE, MAPE, R2, attention-weight extraction); the served forecast and metrics tools currently return placeholder values until a trained checkpoint is wired in
- Built Supabase email-OTP authentication, per-user search history with retention limits, and a background NSE symbol sync (batched upserts, 24-hour freshness check, trigram search)
- Added a pytest suite for the data pipeline, MCP tools, agent orchestration, latency and a market-crash scenario (March 2020); automated deploy to Azure App Service with GitHub Actions
- Repo: [github.com/vrilinda/vyoris_test](https://github.com/vrilinda/vyoris_test)

### Other Project
- **AskiResearchLabs:** FastAPI research-evaluation platform that pulls papers from OpenAlex, CrossRef and arXiv and uses Claude to score AI/ML topics; Python, SQLite, Plotly, JWT auth. [github.com/vrilinda/AskiResearchLabs](https://github.com/vrilinda/AskiResearchLabs)

## Education

**MSc, Artificial Intelligence & Machine Learning** | IIIT Bangalore and Liverpool John Moores University (delivered via upGrad)
Expected 2027 (final project and thesis submitted Dec 2026)

**BSc, Mathematics** | Andhra University, Visakhapatnam | 2005
