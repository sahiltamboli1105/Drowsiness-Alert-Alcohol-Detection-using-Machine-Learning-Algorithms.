import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # This command is to Disable Warning....!!!!

MQ_3 = 3
buzzer = 20
GPIO.setup(3, GPIO.IN)
GPIO.setup(20, GPIO.OUT)
while True:
    j=GPIO.input(MQ_3)
    print(j)
    if j==0 :
        print('Smoke Detected!')
        time.sleep(0.5)
        GPIO.output(20, True)
        time.sleep(0.5)
       
       
    else :
        print ('Smoke Not Detected!')
        time.sleep(0.1)
        GPIO.output(20, False)
