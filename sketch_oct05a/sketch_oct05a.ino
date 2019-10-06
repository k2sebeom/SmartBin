#include <Servo.h>

Servo Mserv;

void setup() {
  // put your setup code here, to run once:
  Mserv.attach(9);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()){
    String type = Serial.readStringUntil('\n');
    int angle = type.toInt();
    Mserv.write(angle);
  }
}
