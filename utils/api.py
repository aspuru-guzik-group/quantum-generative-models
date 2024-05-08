import requests
import sqlite3
from datetime import datetime


class RewardAPI:
    def __init__(
        self,
        username,
        password,
        base_url="https://rip.chemistry42.com",
        db_path="workflows.db",
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.auth = (self.username, self.password)
        self.db_path = db_path
        self.create_table()

    def create_table(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS workflows (
                     id INTEGER PRIMARY KEY,
                     name TEXT,
                     workflow_uuid TEXT UNIQUE,
                     timestamp TEXT)"""
        )
        conn.commit()
        conn.close()

    def save_workflow_id(self, name, workflow_uuid):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO workflows (name, workflow_uuid, timestamp) VALUES (?, ?, ?)",
            (name, workflow_uuid, timestamp),
        )
        conn.commit()
        conn.close()

    def post_smiles(self, name, mpo_score_definition_id, smiles_list):
        url = f"{self.base_url}/v1/score/smiles"
        payload = {
            "mpo_score_definition_id": mpo_score_definition_id,
            "smiles": smiles_list,
        }
        response = requests.post(url, json=payload, auth=self.auth, verify=False)
        print(response)
        if response.status_code == 200:
            workflow_uuid = response.json()["workflow_uuid"]
            self.save_workflow_id(name, workflow_uuid)
            return workflow_uuid
        else:
            raise ValueError("Error posting smiles")

    def get_workflow_results(self, workflow_uuid):
        url = f"{self.base_url}/v1/workflows/{workflow_uuid}/result"
        response = requests.get(url, auth=self.auth, verify=False)
        if response.status_code == 200:
            return response.json()["results"]
        elif response.status_code == 404:
            raise ValueError("Workflow UUID does not exist")
        else:
            raise ValueError("Error getting workflow results")

    def parse_results(self, results, model_name):
        parsed_results = []
        for result in results:
            parsed_result = {
                "smiles": result["smiles"],
                "main_reward": result["main_reward"],
                "filters_passed": result["filters_passed"],
                "ROMol_was_valid": result["ROMol_was_valid"],
                "model": model_name,
            }
            parsed_results.append(parsed_result)
        return parsed_results

    def get_all_workflows(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, workflow_uuid, timestamp FROM workflows")
        result = c.fetchall()
        conn.close()
        return [
            {"id": row[0], "name": row[1], "workflow_uuid": row[2], "timestamp": row[3]}
            for row in result
        ]

    def get_workflow_status(self, workflow_uuid):
        url = f"{self.base_url}/v1/workflows/{workflow_uuid}"
        response = requests.get(url, auth=self.auth, verify=False)
        if response.status_code == 200:
            if response.json()["state"] == "success":
                return response.json()["state"]
            else:
                return response.json()
        elif response.status_code == 404:
            raise ValueError("Workflow UUID does not exist")
        else:
            raise ValueError("Error getting workflow status")

