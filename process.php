<?php
  $sex = $_POST['sex'];
  $bp = $_POST['bp'];
  $cholesterol = $_POST['cholesterol'];
  $age = $_POST['age'];
  $na_to_k = $_POST['na_to_k'];
  
  // Encode categorical variables
  $sex_encoded = ($sex === 'F') ? 1 : 0;
  $bp_high_encoded = ($bp === 'HIGH') ? 1 : 0;
  $bp_low_encoded = ($bp === 'LOW') ? 1 : 0;
  $bp_normal_encoded = ($bp === 'NORMAL') ? 1 : 0;
  $cholesterol_high_encoded = ($cholesterol === 'HIGH') ? 1 : 0;
  $cholesterol_normal_encoded = ($cholesterol === 'NORMAL') ? 1 : 0;
  
  // Encode age using binned categories
  if ($age < 20) {
    $age_binned = '<20s';
  } elseif ($age < 30) {
    $age_binned = '20s';
  } elseif ($age < 40) {
    $age_binned = '30s';
  } elseif ($age < 50) {
    $age_binned = '40s';
  } elseif ($age < 60) {
    $age_binned = '50s';
  } elseif ($age < 70) {
    $age_binned = '60s';
  } else {
    $age_binned = '>60s';
  }
  
  // Encode Na_to_K using binned categories
  if ($na_to_k < 10) {
    $na_to_k_binned = '<10';
  } elseif ($na_to_k < 20) {
    $na_to_k_binned = '10-20';
  } elseif ($na_to_k < 30) {
    $na_to_k_binned = '20-30';
  } else {
    $na_to_k_binned = '>30';
  }
  
  // Prepare input data for prediction
  $data = array(
    'Sex_F' => $sex_encoded,
    'Sex_M' => 1 - $sex_encoded,
    'BP_HIGH' => $bp_high_encoded,
    'BP_LOW' => $bp_low_encoded,
    'BP_NORMAL' => $bp_normal_encoded,
    'Cholesterol_HIGH' => $cholesterol_high_encoded,
    'Cholesterol_NORMAL' => $cholesterol_normal_encoded,
    'Age_binned_<20s' => ($age_binned === '<20s') ? 1 : 0,
    'Age_binned_20s' => ($age_binned === '20s') ? 1 : 0,
    'Age_binned_30s' => ($age_binned === '30s') ? 1 : 0,
    'Age_binned_40s' => ($age_binned === '40s') ? 1 : 0,
    'Age_binned_50s' => ($age_binned === '50s') ? 1 : 0,
    'Age_binned_60s' => ($age_binned === '60s') ? 1 : 0,
    'Age_binned_>60s' => ($age_binned === '>60s') ? 1 : 0,
    'Na_to_K_binned_<10' => ($na_to_k_binned === '<10') ? 1 : 0,
    'Na_to_K_binned_10-20' => ($na_to_k_binned === '10-20') ? 1 : 0,
    'Na_to_K_binned_20-30' => ($na_to_k_binned === '20-30') ? 1 : 0,
    'Na_to_K_binned_>30' => ($na_to_k_binned === '>30') ? 1 : 0
  );
  
  // Make a POST request to the API endpoint for drug prediction
  $ch = curl_init();
  curl_setopt($ch, CURLOPT_URL, 'https://api.example.com/predict');  // Replace with your API endpoint
  curl_setopt($ch, CURLOPT_POST, true);
  curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
  curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
  curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
  $response = curl_exec($ch);
  curl_close($ch);
  
  // Process the response
  $prediction = json_decode($response, true)['prediction'];
  
  // Return the predicted drug
  echo $prediction;
?>
