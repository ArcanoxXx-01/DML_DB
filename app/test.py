import time, requests
from pathlib import Path


DATASETS = Path("./datasets")


# --- TEST DATASETS ---
def test_upload_dataset():
    csv_content = "a,b,c\n1,2,3\n4,5,6"
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/datasets",
        files={"file": ("test.csv", csv_content)},
        data={"dataset_id": "ds1"},
    )
    assert resp.status_code == 200
    assert "batches" in resp.json()
    print("Create Dataset: OK")
    print(resp.json())


def test_get_dataset_batch():
    resp = requests.get("http://127.0.0.1:8000/api/v1/datasets/ds1/0")
    assert resp.status_code == 200
    print("GET Batch: OK")
    print(resp)


def test_list_datasets():
    resp = requests.get("http://127.0.0.1:8000/api/v1/datasets/list")
    assert "ds1" in resp.json()
    print("GET Datasets: OK")
    print(resp.json())


# --- TEST TRAININGS ---
def test_create_training():
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/trainings",
        data={
            "training_id": "t1",
            "dataset_id": "ds1",
            "training_type": "regression",
            "models_names": ["m1", "m2"],
        },
    )
    assert resp.status_code == 200
    print("Create Training: OK")
    print(resp.json())


def test_get_training():
    resp = requests.get("http://127.0.0.1:8000/api/v1/trainings/t1")
    assert resp.status_code == 200
    assert resp.json()["training_id"] == "t1"
    print("GET Training: OK")
    print(resp.json())


# --- TEST MODELS ---
def test_update_health():
    # get a model
    t = requests.get("http://127.0.0.1:8000/api/v1/trainings/t1").json()
    model_id = t["models_id"][0]

    resp = requests.put(f"http://127.0.0.1:8000/api/v1/models/health/{model_id}")
    assert resp.status_code == 200
    print("MODEL Health: OK")
    print(resp.json())


def test_model_to_run():
    time.sleep(1)
    resp = requests.get("http://127.0.0.1:8000/api/v1/models/torun")
    assert resp.status_code == 200
    print("MODEL To Run: OK")
    print(resp.json())


def test_model_metrics_update_and_get():
    t = requests.get("http://127.0.0.1:8000/api/v1/trainings/t1").json()
    model_id = t["models_id"][0]
    print(model_id)

    update = requests.put(
        f"http://127.0.0.1:8000/api/v1/models/{model_id}",
        data={"results": ["0.1", "0.2"]},
    )
    assert update.status_code == 200
    print(update.json())

    getm = requests.get(f"http://127.0.0.1:8000/api/v1/models/{model_id}")
    assert getm.status_code == 200
    assert getm.json()["model_data"] == [
        "0.1",
        "0.2",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]
    print(getm.json())


def run_tests():
    print("\n\n===== DATASETS TESTS =====\n")
    test_upload_dataset()
    test_get_dataset_batch()
    test_list_datasets()
    print("\n\n===== TRAININGS TESTS =====\n")
    test_create_training()
    test_get_training()
    print("\n\n===== MODELS TESTS =====\n")
    test_update_health()
    test_model_to_run()
    # time.sleep(20)
    # test_model_to_run()
    # test_model_metrics_update_and_get()


if __name__ == "__main__":
    run_tests()
