const int triggerPin = 4;  // Digital pin D4
const int threshold = 1; // Threshold (e.g., HIGH signal)
unsigned long lastTriggerTime = 0; // Last time the trigger was sent
unsigned long debounceDelay = 2000; // Delay to ignore further signals for 1 second (1000 ms)

void setup() {
  pinMode(triggerPin, INPUT);  // Set pin D4 as an input
  Serial.begin(9600);  // Start serial communication
}

void loop() {
  // Read the digital value from pin D4
  if (digitalRead(triggerPin) == threshold) {
    unsigned long currentMillis = millis(); // Get the current time in milliseconds
    
    // Check if enough time has passed since the last trigger
    if (currentMillis - lastTriggerTime >= debounceDelay) {
      Serial.println("O"); // Send trigger signal when the threshold is exceeded
      lastTriggerTime = currentMillis; // Update the last trigger time
      //Serial.flush();
    }
  }
}
