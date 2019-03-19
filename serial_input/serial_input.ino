int input = 0;
void setup() {
  // put your setup code here, to run once:
   pinMode(13, OUTPUT);
   Serial.begin(115200);
   digitalWrite(13, LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
   if (Serial.available() > 0) {
    input = Serial.read();
    Serial.print("I received: ");
    Serial.println(input);
    if (input == 1) digitalWrite(13, HIGH);
    else if (input == 0) digitalWrite(13, LOW);
   }
}
