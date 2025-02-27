const int triggerPin = 4;  // Digital pin D4
const int counterPin = 7;  // Digital pin D4
const int threshold = 1; // Threshold (e.g., HIGH signal)
unsigned long lastTriggerTime = 0; // Last time the trigger was sent
unsigned long debounceDelay = 2000; // Delay to ignore further signals for 1 second (1000 ms)
int state = 0;
int state_count = 0;


int val;
int counter_read;


void setup() {
  pinMode(triggerPin, INPUT);  // Set pin D4 as an input
  Serial.begin(9600);  // Start serial communication
  state = 0;
}

void loop() {
  // Read the digital value from pin D4
  val=digitalRead(triggerPin);
  counter_read=digitalRead(counterPin);

  if (state == 0 && val == 1){
    state = 1;
    Serial.println("O");
  }
  else if (state == 1 && val == 0){
    state = 0;
  }

  if (state_count == 0 && counter_read == 1){
    state_count = 1;
    Serial.println("C");
  }
  else if (state_count == 1 && counter_read == 0){
    state_count = 0;
  }
  
  
  /*if (digitalRead(triggerPin) == threshold) {
    unsigned long currentMillis = millis(); // Get the current time in milliseconds
    
    // Check if enough time has passed since the last trigger
    if (currentMillis - lastTriggerTime >= debounceDelay) {
      Serial.println("O"); // Send trigger signal when the threshold is exceeded
      lastTriggerTime = currentMillis; // Update the last trigger time
      //Serial.flush();
    }
  }*/
}
