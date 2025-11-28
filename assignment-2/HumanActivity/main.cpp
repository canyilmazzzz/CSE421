#include "mbed.h"

extern "C" {
    #include "bayes_har_config.h"
}

#include <cmath>

#define TIME_PERIODS  80
#define N_FEATURES    NUM_FEATURES

void compute_features(const float x[TIME_PERIODS],
                      const float y[TIME_PERIODS],
                      const float z[TIME_PERIODS],
                      float features[N_FEATURES])
{
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    for (int i = 0; i < TIME_PERIODS; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_z += z[i];
    }
    float mean_x = sum_x / TIME_PERIODS;
    float mean_y = sum_y / TIME_PERIODS;
    float mean_z = sum_z / TIME_PERIODS;

    float var_x = 0.0f, var_y = 0.0f, var_z = 0.0f;
    float min_x = x[0], min_y = y[0], min_z = z[0];
    float max_mag = 0.0f;

    for (int i = 0; i < TIME_PERIODS; i++) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        float dz = z[i] - mean_z;

        var_x += dx * dx;
        var_y += dy * dy;
        var_z += dz * dz;

        if (x[i] < min_x) min_x = x[i];
        if (y[i] < min_y) min_y = y[i];
        if (z[i] < min_z) min_z = z[i];

        float mag = std::sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
        if (mag > max_mag) max_mag = mag;
    }

    float std_x = std::sqrt(var_x / TIME_PERIODS);
    float std_y = std::sqrt(var_y / TIME_PERIODS);
    float std_z = std::sqrt(var_z / TIME_PERIODS);

    features[0] = mean_x;
    features[1] = mean_y;
    features[2] = mean_z;
    features[3] = std_x;
    features[4] = std_y;
    features[5] = std_z;
    features[6] = min_x;
    features[7] = min_y;
    features[8] = min_z;
    features[9] = max_mag;
}

int bayes_predict(const float features[N_FEATURES])
{
    int best_class = 0;
    float best_score = -1e30f;

    for (int c = 0; c < NUM_CLASSES; c++) {

        float diff[N_FEATURES];
        for (int i = 0; i < N_FEATURES; i++) {
            diff[i] = features[i] - MEANS[c][i];
        }

        float quad = 0.0f;
        for (int i = 0; i < N_FEATURES; i++) {
            for (int j = 0; j < N_FEATURES; j++) {
                quad += diff[i] * INV_COVS[c][i][j] * diff[j];
            }
        }

        float prior = CLASS_PRIORS[c];
        if (prior < 1e-12f) prior = 1e-12f;

        float det = DETS[c];
        if (det < 1e-12f) det = 1e-12f;

        float score = std::log(prior) - 0.5f * (std::log(det) + quad);

        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    return best_class;
}

const char* CLASS_NAMES[NUM_CLASSES] = {
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking"
};

int main()
{
    static float x[TIME_PERIODS];
    static float y[TIME_PERIODS];
    static float z[TIME_PERIODS];

    for (int i = 0; i < TIME_PERIODS; i++) {
        x[i] = 0.0f;
        y[i] = 0.0f;
        z[i] = 9.81f;
    }

    float features[N_FEATURES];
    compute_features(x, y, z, features);

    int pred_class = bayes_predict(features);
    const char* activity = CLASS_NAMES[pred_class];

    printf("Predicted activity: %s (class %d)\r\n", activity, pred_class);

    while (true) {
        thread_sleep_for(500);
    }
}
