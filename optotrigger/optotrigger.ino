
const int digitalPin = 4;  
const int daqPin = 5;   

const String CODE1 = "a";
const String CODE2 = "b";
const String CODE3 = "c";
const String CODE8 = "h";
const String CODE9 = "i";
const String CODE10 = "j";
bool triggered = false;
bool lastState = false;
bool identified = false;

String receivedCode = "";

void setup() {
  Serial.begin(115200);     
  Serial1.begin(115200);   
  pinMode(digitalPin, INPUT); 
  pinMode(daqPin, OUTPUT); 
  digitalWrite(daqPin, LOW);
}

bool isValidCode(String code) {
  return (code == CODE1 || code == CODE2 || code == CODE3 || code == CODE8 || code == CODE9 || code == CODE10);
}

void loop() {

  if (!identified && !triggered ){
      while (Serial1.available() > 0) {
        int c = Serial1.read();
        if (c>90){
          Serial.println(c);
          Serial1.flush();
          identified=true;

          c=0;
          }
        }
      }

  bool currentState = digitalRead(digitalPin);
  if (!triggered && identified && currentState >0.5) {
    Serial.println(currentState);
    digitalWrite(daqPin, HIGH);
    triggered = true;


  }
  else if (triggered && identified && currentState <0.5) {
    Serial.println(currentState);
    digitalWrite(daqPin, LOW);
    identified = false;
    triggered = false;



  }

  
}
