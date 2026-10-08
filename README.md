# ANDM Self-Evolving RL Streamlit System

This folder contains the Streamlit demonstration system for the controlled ONOS–Mininet research experiment inspired by Alfred Nzo District Municipality.

The Streamlit system does not replace the notebook. The notebook contains the training and research workflow. Streamlit presents the measured evidence and the final saved results in a faster and clearer form for the system demonstration.

## Start the system on Windows

1. Extract the complete ZIP folder.
2. Open the extracted `ANDM_Streamlit_System` folder.
3. Double-click `INSTALL_WINDOWS.bat` the first time only.
4. Wait until all packages have installed.
5. Double-click `START_WINDOWS.bat`.
6. The system should open in your browser at `http://localhost:8501`.

If the browser does not open automatically, copy `http://localhost:8501` into Chrome or Edge.

## Start the system from Command Prompt

Open Command Prompt inside this folder and run:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Start the system on Ubuntu

Run:

```bash
cd ANDM_Streamlit_System
chmod +x install_ubuntu.sh start_ubuntu.sh
./install_ubuntu.sh
./start_ubuntu.sh
```

## Pages in the system

- **Overview:** Introduces the research, network profiles and ANDM-inspired topology.
- **Measured network:** Shows CPU, throughput, latency, packet loss and bar graphs with simple explanations.
- **Agent comparison:** Shows the ten-seed DQN and PPO comparison and the selection decision.
- **Self-evolving cycle:** Replays the stress stream, ADWIN drift detection and candidate promotion.
- **Final decision:** Compares deployed DQN v2, fixed allocation and the simple CPU rule.
- **Evidence and sources:** Checks the SHA-256 hashes and provides the recent supporting research.

## Folder structure

```text
ANDM_Streamlit_System/
├── app.py
├── Self_Evolving_RL_ANDM_Measured_Mininet.ipynb
├── requirements.txt
├── INSTALL_WINDOWS.bat
├── START_WINDOWS.bat
├── data/
│   ├── mininet_baseline_cpu.csv
│   ├── mininet_baseline_network_raw.csv
│   ├── mininet_rural_cpu.csv
│   └── mininet_rural_network_raw.csv
├── evidence/
│   ├── andm_mininet_capture_final_v5.py
│   └── final_evidence_sha256.txt
├── results/
│   ├── algorithm_summary.csv
│   ├── drift_log.csv
│   ├── promotion_validation_summary.csv
│   └── final_rural_test_summary.csv
└── assets/
    └── saved notebook figures
```

## Research boundaries

- The data came from a controlled ONOS–Mininet testbed, not the real Alfred Nzo network.
- Morning, Afternoon and Night are simulated workload periods.
- The 20 Mbps bandwidth, 50 ms delay and 2% loss are researcher-defined rural test settings.
- The RL environment makes simulated vCPU decisions. It does not physically attach or remove CPU cores.
- The deployed DQN improved after adaptation, but the simple CPU rule achieved the highest final reward.
