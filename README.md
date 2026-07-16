# RansKnow: Ransomware Knowledge Dataset

A curated dataset of **761 video records** (718 with full transcripts) sourced from **50 cybersecurity YouTube channels**, built to support knowledge extraction, NLP research, and threat intelligence analysis focused on ransomware.

---

## Background

Ransomware attacks are one of the most damaging categories of cyber threats, yet knowledge about tactics, tools, and affected platforms is scattered across conference talks, incident response reports, and security briefings. RansKnow systematically collects and structures this knowledge from publicly available video content.

Videos were selected using keyword-based inclusion criteria (ransomware-relevant terms), filtered for technical depth, and processed through a Knowledge Agent pipeline to extract structured features aligned with the MITRE ATT&CK framework.

---

## Repository Structure

```
RansKnow/
├── transcripts/                  # 761 video records across 50 channels
│   ├── C01_The_DFIR_Report/
│   │   ├── V0001.txt             # Raw transcript (YouTube timestamped format)
│   │   ├── V0001.meta.json       # Video metadata
│   │   └── ...
│   ├── C02_SANS_Digital_Forensics_and_Incident_Response/
│   └── ... (C01–C50)
│
├── outputs/                      # Knowledge Agent extraction results
│   ├── Knowledge_Agent_Features_718.csv  # v2 — full 718-row feature CSV (current)
│   ├── Knowledge_Agent_Features_718.xlsx
│   ├── Knowledge_Agent_Features_307.csv  # v1 — original 307-row CSV (historical)
│   ├── Knowledge_Agent_Features_307.xlsx
│   ├── Knowledge_Agent_Output.csv
│   └── Knowledge_Agent_Uncertainty_Distribution.png
│
├── RansKnow_v1/
│   └── Knowledge_Agent_Features_v1.csv   # v1 feature CSV (307 rows)
│
├── Scripts/                      # Data pipeline notebooks and scripts
│   ├── knowledge_agent.py                # v2 feature extraction pipeline (current)
│   ├── fetch_transcripts.py              # Transcript fetcher with Whisper ASR fallback
│   ├── ransknow-getting-started.ipynb    # Getting Started notebook (Kaggle-ready)
│   ├── 01_Transcript_Dataset_Construction.ipynb
│   ├── fetch_transcripts_keywords.ipynb
│   ├── Data_extraction_inclusion.ipynb
│   ├── Inclusin_Criteria_2.ipynb
│   ├── Knowledge_Agent_Modelling.ipynb
│   ├── fill_video_selection_rubric_openpyxl.py
│   └── populate_video_registry_from_transcripts.py
│
├── rubrics/                      # Scoring rubrics and channel registry
│   ├── Dataset_Channel_Registry_Batch2.xlsx  # Batch 2 candidates — C51–C101 (51 channels, pending)
│   ├── Dataset_Channel_Registry_Populated_25.xlsx
│   ├── Dataset_Channel_Registry_Updated_50_fixed_urls.xlsx
│   ├── Ransomware_Family_Coverage_List.xlsx
│   └── Video_Selection_Rubric_*.xlsx
│
├── Figures/                      # Visualisations
├── Progress_Mapping/             # Weekly progress tracking
├── Channel_Registry_1.xlsx       # Master registry of all 50 channels
├── Family_Coverage_Targets_Rules.docx
├── Ransomware_Transcript_Dataset_Summary.pdf
└── Video_Selection_Rubric_AutoScore_Filled_6.xlsx
```

---

## Knowledge Agent Feature Schema

All **718 videos** with real transcripts have been processed through the Knowledge Agent pipeline (`Scripts/knowledge_agent.py`), producing **37 structured feature columns**:

| Category | Columns |
|---|---|
| **Identifiers** | `Video_ID`, `Channel_ID`, `Channel_Name`, `Video_Title`, `YouTube_URL`, `Transcript_Path` |
| **Metadata** | `Year`, `DurationSeconds`, `Transcript_Provider` |
| **Ransomware Families** | `Family_Count`, `Family_List` |
| **MITRE ATT&CK Tactics** | `Tactic_Initial_Access`, `Tactic_Execution`, `Tactic_Persistence`, `Tactic_Privilege_Escalation`, `Tactic_Credential_Access`, `Tactic_Lateral_Movement`, `Tactic_Discovery`, `Tactic_Command_and_Control`, `Tactic_Exfiltration`, `Tactic_Impact`, `Tactic_Total_Mentions`, `Dominant_Tactic` |
| **Tools** | `Tool_Cobalt_Strike`, `Tool_Mimikatz`, `Tool_PsExec`, `Tool_Rclone`, `Tool_MegaNZ`, `Tool_AnyDesk`, `Tool_TeamViewer`, `Tool_BloodHound`, `Tool_Total_Mentions`, `Tool_List` |
| **Platforms** | `Platform_Signal`, `Platform_Windows`, `Platform_Linux`, `Platform_ESXi` |

---

## Channels Covered (C01–C50)

| ID | Channel |
|---|---|
| C01 | The DFIR Report |
| C02 | SANS Digital Forensics and Incident Response |
| C03 | Black Hat |
| C04 | DEF CON Conference |
| C05 | CrowdStrike |
| C06 | Mandiant / Google Cloud Security |
| C07 | Sophos X-Ops |
| C08 | Red Canary |
| C09 | Huntress |
| C10 | John Hammond |
| C11 | Secureworks |
| C12 | Kaspersky |
| C13 | Palo Alto Networks Unit 42 |
| C14 | Elastic Security |
| C15 | RSA Conference |
| C16 | USENIX Security |
| C17 | FIRST Conference |
| C18 | Malware Analysis for Hedgehogs |
| C19 | OALabs |
| C20 | LiveOverflow |
| C21 | CyberWire |
| C22 | Threatpost |
| C23 | Microsoft Security |
| C24 | Cisco Talos Intelligence Group |
| C25 | Recorded Future |
| C26 | MalwareTech |
| C27 | VX-Underground |
| C28 | Black Hills Information Security |
| C29 | Security Onion Solutions |
| C30 | SANS Institute |
| C31 | Mandiant |
| C32 | Sophos |
| C33 | ESET |
| C34 | Elastic |
| C35 | Splunk |
| C36 | Wazuh |
| C37 | TrustedSec |
| C38 | Blue Team Village |
| C39 | MITRE ATT&CK |
| C40 | Magnet Forensics |
| C41 | Belkasoft |
| C42 | DFIR Science |
| C43 | Active Countermeasures |
| C44 | VMware Carbon Black |
| C45 | Arctic Wolf |
| C46 | Dragos Inc |
| C47 | Cybereason |
| C48 | ThreatLocker |
| C49 | LogRhythm |
| C50 | Darktrace |

Most channels contribute **10 videos** each, selected to maximise ransomware family coverage and technical depth.

### Batch 2 — Planned Expansion (C51–C101)

A second batch of **51 candidate channels** (C51–C101) has been curated to expand coverage to ~100 channels, targeting gaps in the current dataset:

| Category | Channels |
|---|---|
| Threat Intelligence / Vendor | SentinelOne, Check Point Research, Trend Micro, WithSecure, Rapid7, Malwarebytes, Group-IB, Prodaft, Anomali, Kroll, NCC Group, Emsisoft, Intezer, ZeroFox, ReliaQuest, Binary Defense |
| DFIR Practitioners | 13Cubed, HuskyHacks, TCM Security, Josh Stroschein, ANY.RUN, Cado Security, Cyb3rWard0g |
| Conferences | Virus Bulletin, Hack In The Box, Wild West Hackin' Fest, SecTor, CactusCon, GrayHat, ICS Village, Hack.lu, Troopers, BlueHat, CYBERWARCON, OffensiveCon |
| OT / ICS / Government | Claroty, Nozomi Networks, S4 Conference, CISA, Waterfall Security |
| News & Analysis | Risky Business, Security Weekly, Darknet Diaries |
| Blue Team / Detection | Florian Roth, Eric Zimmerman, LetsDefend |
| Consulting | GuidePoint Security, ThreatConnect, Optiv, CanSecWest |

Channel selection rubric and URLs: `rubrics/Dataset_Channel_Registry_Batch2.xlsx`

---

## Transcript Format

Each `.txt` file contains the raw YouTube transcript in timestamped format:

```
0:00
Welcome to the DFIR report...
0:05
Today we are covering a LockBit intrusion...
```

Each `.meta.json` file contains structured metadata:

```json
{
  "Video_ID": "V0001",
  "Channel_ID": "C01",
  "Channel_Name": "The DFIR Report",
  "Channel_UC": "UC6R2MPMkkCqFxvAdQAI_23A",
  "YouTube_Video_ID": "xxxxxx",
  "Video_Title": "...",
  "Year": 2024,
  "PublishedAt": "2024-03-15T14:00:00Z",
  "DurationSeconds": 1843,
  "DurationISO": "PT30M43S",
  "Matched_Keywords": "ransomware;ir",
  "Transcript_Available": true
}
```

---

## Intended Use Cases

- Ransomware threat intelligence research
- NLP and information extraction on cybersecurity text
- MITRE ATT&CK tactic classification
- Tool and platform attribution modelling
- Knowledge graph construction for ransomware
- Curriculum development for cybersecurity education

---

## Dataset on Kaggle

The full dataset (including transcript zips, scripts, and rubrics) is also available on Kaggle:
[https://www.kaggle.com/datasets/henrykabuye/ransknow-v1](https://www.kaggle.com/datasets/henrykabuye/ransknow-v1)

A **Getting Started notebook** is available on the dataset's Code tab, covering feature exploration, tactic/tool plots, transcript reading, and family filtering:
[https://www.kaggle.com/code/henrykabuye/ransknow-getting-started](https://www.kaggle.com/code/henrykabuye/ransknow-getting-started)

The notebook source is also included at `Scripts/ransknow-getting-started.ipynb`.

---

## Version History

| Version | Date | Description |
|---|---|---|
| v1 | May 2026 | Initial release — Knowledge Agent CSV (307 rows) |
| v2 | July 2026 | Automated fetch — 761 video records, 424 transcripts, all scripts, rubrics, figures, and documentation |
| v3 | July 2026 | Whisper ASR fallback — 718 transcripts recovered (94% coverage) across all 761 video records |
| v4 | July 2026 | Knowledge Agent v2 — 718-row feature CSV (`Knowledge_Agent_Features_718.csv`), fixed tool detection, added Year/DurationSeconds/Transcript_Provider columns |

---

## License

This dataset is released under the [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) licence.
