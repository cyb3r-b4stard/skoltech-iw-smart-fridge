#include <GxEPD2_BW.h>
#include <Adafruit_GFX.h>

// Sensor Pin Definitions
const int MQ3_PIN   = A3;
const int MQ5_PIN   = A5;
const int MQ135_PIN = A0;

// FIX 1: Change buffer height from full screen (200px) to a page slice (32px).
// This reduces RAM usage from >5KB down to a few hundred bytes.
GxEPD2_BW<GxEPD2_154_D67, 32> display(GxEPD2_154_D67(/*CS=*/ 10, /*DC=*/ 9, /*RST=*/ 8, /*BUSY=*/ 7));

void setup() {
  Serial.begin(9600);

  // Initialize display
  display.init(115200);
  display.setRotation(1); // Set orientation (0 to 3)
}

void loop() {
  // Read raw analog values (0 to 1023)
  int mq3_val   = analogRead(MQ3_PIN);
  int mq5_val   = analogRead(MQ5_PIN);
  int mq135_val = analogRead(MQ135_PIN);

  // Log values to Serial Monitor
  Serial.print(F("MQ-3: ")); Serial.print(mq3_val);
  Serial.print(F(" | MQ-5: ")); Serial.print(mq5_val);
  Serial.print(F(" | MQ-135: ")); Serial.println(mq135_val);

  // Update e-Paper Display using page-by-page rendering
  drawSensorValues(mq3_val, mq5_val, mq135_val);

  // Delay before next refresh (e-Paper shouldn't refresh too rapidly)
  delay(10000); 
}

void drawSensorValues(int mq3, int mq5, int mq135) {
  display.setFullWindow();
  
  // The firstPage/nextPage loop handles page-by-page rendering automatically
  display.firstPage();
  do {
    display.fillScreen(GxEPD_WHITE);
    display.setTextColor(GxEPD_BLACK);

    // FIX 2: Wrap static strings in F() macro to store them in Flash memory instead of RAM
    display.setTextSize(2);
    display.setCursor(10, 10);
    display.print(F("GAS MONITOR"));

    display.drawFastHLine(0, 32, 200, GxEPD_BLACK);

    // MQ-3
    display.setTextSize(1);
    display.setCursor(10, 45);
    display.print(F("MQ-3 (Alcohol):"));
    display.setTextSize(2);
    display.setCursor(10, 58);
    display.print(mq3);

    // MQ-5
    display.setTextSize(1);
    display.setCursor(10, 85);
    display.print(F("MQ-5 (LPG/Gas):"));
    display.setTextSize(2);
    display.setCursor(10, 98);
    display.print(mq5);

    // MQ-135
    display.setTextSize(1);
    display.setCursor(10, 125);
    display.print(F("MQ-135 (Air Q):"));
    display.setTextSize(2);
    display.setCursor(10, 138);
    display.print(mq135);

  } while (display.nextPage());
}