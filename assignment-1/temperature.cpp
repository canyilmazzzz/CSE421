#include "mbed.h"
#include "stm32f7xx_hal.h"
#include "stm32746g_discovery_lcd.h"
#include "stm32746g_discovery_ts.h"

FileHandle *mbed::mbed_override_console(int){
    static UnbufferedSerial pc(USBTX, USBRX, 115200);
    return &pc;
}

ADC_HandleTypeDef hadc1;
static void adc_init() {
    __HAL_RCC_ADC1_CLK_ENABLE();
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCKPRESCALER_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode = DISABLE;
    hadc1.Init.ContinuousConvMode = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    hadc1.Init.DMAContinuousRequests = DISABLE;
    hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    HAL_ADC_DeInit(&hadc1);
    HAL_ADC_Init(&hadc1);

    ADC_ChannelConfTypeDef sConfig = {0};
    sConfig.Channel = ADC_CHANNEL_TEMPSENSOR;
    sConfig.Rank = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}

static uint16_t adc_get_raw_value() {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 10);
    uint16_t raw_value = (uint16_t)HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    return raw_value;
}

static float convert_to_celsius(uint16_t adc_raw_value){
    const float VREF = 3.3f;
    const float ADC_MAX = 4095.0f;
    const float V25 = 0.76f;
    const float Avg_Slope = 0.0025f;

    float Vsense = (adc_raw_value * VREF) / ADC_MAX;
    float celsius_value = ((Vsense - V25) / Avg_Slope) + 25.0f;

    return celsius_value;
}

int main() {
    BSP_LCD_Init();
    BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS);
    BSP_LCD_SelectLayer(0);
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetBackColor(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
    BSP_LCD_DisplayStringAt(0, 10, (uint8_t *)"Temperature Measurements", CENTER_MODE);
    
    ThisThread::sleep_for(100ms);
    adc_init();

    int y = 50;
    for(int i = 0; i < 10; i++){
        uint16_t raw_adc = adc_get_raw_value();
        float celsius_value = convert_to_celsius(raw_adc);

        char buffer[64];
        sprintf(buffer, "Temp. %d: %.2f C", i+1, celsius_value);
        BSP_LCD_DisplayStringAt(0, y, (uint8_t*)buffer, CENTER_MODE);
        y += 20;
    }
}