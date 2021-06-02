This documentation will help you to start controlling the LINAK table-leg

Prerequisites:
python3
pip3

Requirement:

1. Install libusb1 library:
pip3 install libusb1

2. Create udev rule file to give a user access to Linak USB device (or copy/paste file "10-linak-libusb.rules" from this folder to correct one)
cd /etc/udev/rules.d
sudo touch 10-linak-libusb.rules
sudo nano 10-linak-libusb.rules

3. Write a rule into created file, close and save it.
SUBSYSTEM=="usb", ATTR{idVendor}=="12d3", ATTR{idProduct}=="0002", MODE="0666"

4. In terminal, refresh udev to enable new rule:
sudo udevadm trigger

Using LINAK:

Connect the USB2LIN cable from your computer to the control box.

Type lsusb in terminal to see if the USB is recognized. Here you will also get the Product ID and Vendor ID that will be be used in the code, hopefully the ID's will not be changed with a new product and nothing will have to be changed.

Clean up code and make is as usable as possible for us
See if there is any issue using other peoples "Base" code, if yeah... the re-write from the start but it will be basically the same...  
