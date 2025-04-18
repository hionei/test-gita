from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.reward import compute_softmax
from synth.validator.price_data_provider import PriceDataProvider
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import os
from datetime import datetime, timedelta
from synth.utils.helpers import (
    convert_prices_to_time_format,
)

def test_calculate_crps_for_miner_1(real_path, miner_paths):
    time_increment = 300  # 300 seconds = 5 minutes

    sum_all_scores, _ = calculate_crps_for_miner(
        np.array(miner_paths),
        np.array(real_path),
        time_increment,
    )
    return sum_all_scores

def test_normalization(arr):
    result = compute_softmax(np.array(arr))
    return result

def test_generate_simulations(sigma, start_time, current_value):
    result = generate_simulations(
        asset="BTC",
        start_time=start_time,
        time_increment=300,
        time_length=86400,
        num_simulations=100,
        current_value=current_value,
        sigma=sigma
    )
    return result


def generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=1,
    sigma=0.0035,
    current_value=87654.95502634
):
    if start_time is None:
        raise ValueError("Start time must be provided.")

    current_price = current_value
    if current_price is None:
        raise ValueError(f"Failed to fetch current price for asset: {asset}")

    simulations = simulate_crypto_price_paths_origin(
        current_price=current_price,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        sigma=sigma,
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    return predictions


def simulate_single_price_path_origin(
    current_price, time_increment, time_length, sigma
):
    """
    Simulate a single crypto asset price path.
    """
    one_hour = 3600
    dt = time_increment / one_hour
    num_steps = int(time_length / time_increment)
    std_dev = sigma * np.sqrt(dt)
    price_change_pcts = np.random.normal(0, std_dev, size=num_steps)
    cumulative_returns = np.cumprod(1 + price_change_pcts)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    price_path = current_price * cumulative_returns
    return price_path


def simulate_crypto_price_paths_origin(
    current_price, time_increment, time_length, num_simulations, sigma
):
    """
    Simulate multiple crypto asset price paths.
    """

    price_paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path_origin(
            current_price, time_increment, time_length, sigma
        )
        price_paths.append(price_path)

    return np.array(price_paths)


# def generate_simulations(
#     asset="BTC",
#     start_time=None,
#     time_increment=300,
#     time_length=86400,
#     num_simulations=1,
#     sigma=0.0035,
#     current_value=87654.95502634
# ):
#     if start_time is None:
#         raise ValueError("Start time must be provided.")

#     current_price = current_value
#     if current_price is None:
#         raise ValueError(f"Failed to fetch current price for asset: {asset}")

#     simulations = simulate_crypto_price_paths(
#         current_price=current_price,
#         time_increment=time_increment,
#         time_length=time_length,
#         num_simulations=num_simulations,
#         sigma=sigma,
#     )

#     predictions = convert_prices_to_time_format(
#         simulations.tolist(), start_time, time_increment
#     )

#     return predictions

# def simulate_single_price_path(
#     current_price, time_increment, time_length, growth_rate, transition_ratio
# ):
#     num_steps = int(time_length / time_increment)
#     transition_steps = int(num_steps * transition_ratio)
#     linear_steps = num_steps - transition_steps
    
#     # Step 1: Create the rounded curve (using a quadratic function)
#     t1 = np.linspace(0, 1, num_steps)  # 0 to 1 range for first phase
#     rounded_curve = 1 + (growth_rate - 1) * np.log(t1 * 30 + 1) # Sigmoid curve

#     full_growth = np.insert(rounded_curve, 0, 1.0)

#     # Generate final price path
#     price_path = current_price * full_growth

#     return price_path

# def simulate_crypto_price_paths(
#     current_price, time_increment, time_length, num_simulations, sigma
# ):
#     price_paths = []
    
#     min_growth_rate = 1 - sigma
#     max_growth_rate = 1 + sigma
    
#     growth_rates = np.linspace(min_growth_rate, max_growth_rate, num_simulations)
    
#     transition_ratio = 0.2
    
#     for rate in growth_rates:
#         price_path = simulate_single_price_path(
#             current_price, time_increment, time_length, rate, transition_ratio
#         )
#         price_paths.append(price_path)

#     return np.array(price_paths)

def test_fetch_data(end_point):
    _dataProvider = PriceDataProvider("BTC")

    result = _dataProvider.fetch_data(end_point)
    return result

if __name__ == "__main__":
    end_time = "2025-03-16T03:44:00+00:00"
    dt = datetime.fromisoformat(end_time)
    two_days_ago = dt - timedelta(days=1)
    start_time = two_days_ago.isoformat()
    print(start_time, end_time)
    real_price_path = test_fetch_data(end_time)
    real_path = [entry["price"] for entry in real_price_path]

    #Saving real price path
    base_dir = "./resultData/miner4"
    os.makedirs(base_dir, exist_ok=True)
    real_price_file_path = os.path.join(base_dir, "real_price.json")
    print("Real path length", len(real_path))
    with open(real_price_file_path, "w", encoding="utf-8") as file:
        json.dump(real_path, file,  indent=4)

    arr1 = [round(0.001 + i * 0.00005, 7) for i in range(int((0.01 - 0.001) / 0.00005) + 1)]
    score_array = []

    for index, value in enumerate(arr1, start=1):
        current_value = real_path[0]
        generated_simulations = test_generate_simulations(value, start_time, current_value)
        miner_paths = [[entry['price'] for entry in sublist] for sublist in generated_simulations]
        # print("Miner path length", len(miner_paths[0]))

        #Saving miner path file
        # miner_path_file = os.path.join(base_dir, f"miner_path_{index}.json")

        # with open(miner_path_file, "w", encoding="utf-8") as file:
        #     json.dump({"sigma": value, "miner_paths": miner_paths}, file, indent=4)

        score1 = test_calculate_crps_for_miner_1(real_path, miner_paths)
        score_array.append(score1)
    
    normalzied_scores = test_normalization(score_array)

    score_file_path = os.path.join(base_dir, "scores.json")

    data_dict = {float(k): float(v) for k, v in zip(arr1, normalzied_scores)}
    
    with open(score_file_path, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, indent=4)