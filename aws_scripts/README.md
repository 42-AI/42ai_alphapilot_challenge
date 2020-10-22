## AWS setup

### 1. Flightgoggles and ROS installation
On *both* the AWS instance and the local machine, install the Flightgoggles simulation and the ROS backend by running the following installation script: 
`./aws_scripts/aws-rosinstall.sh`

This script automates the manual work normally required for the installation, which is composed of three main steps:
- [ROS installation (= Kinetic distribution)](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- [Python packages installation, at the bottom of this page](https://github.com/mit-fast/FlightGoggles/wiki/Prerequisites-and-Testing-Setup)
- [Flightgoggles installation](https://github.com/mit-fast/FlightGoggles/wiki/installation-local)

### 2. Network Setup
Once the simulation is installed on *both* the AWS instance and the local machine, some additionnal setup is required to allow the two machines to communicate.
#### OpenVPN installation
Start by setting up a VPN between the instance and the local machine, for example by following [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-set-up-an-openvpn-server-on-ubuntu-16-04).
Careful: in `~/client-configs/base.conf`, use the AWS EC2 public IP in the following line: `remote server_IP_address 1194`.

After running `sudo openvpn --config NOMDUCLIENT.ovpn` on the local machine, and `sudo systemctl start openvpn@server` on AWS instance, the VPN should be up and running. You can obtain a confirmation that the VPN is working with the following commands:
- on the local machine, `sudo openvpn --config NOMDUCLIENT.ovpn` should end with `Initialization Sequence Completed`.
- on AWS, `sudo systemctl status openvpn@server` should display `Active: active (running)`; the name of the client and the time of connection should also appear in the logs.
- Finally, network interfaces 10.8.0.1 (server) and 10.8.0.6 (local machine) should appear in `ifconfig`, and be pingable ! 
#### Ros environment variables
The two machines can communicate through the VPN by using the two network interfaces 10.8.0.1 and 10.8.0.6.
In order to run ROS through the VPN, we need to specify these two addresses in environment variables:
- On the AWS instance, ensure that the variables are set as follows:
```
echo $ROS_MASTER_URI $ROS_IP
http://10.8.0.1:11311 10.8.0.1
```
- On the local machine, ensure that the variables are set as follows::
```
echo $ROS_MASTER_URI $ROS_IP
http://10.8.0.1:11311 10.8.0.6
```
#### AWS security groups and firewall disable
Finally, you need to configure the security groups and the firewall; these two tools have basically the same function, the former operates at the AWS level (traffic never reach the instance if port is blocked) and the later operates at the instance (virtual machine) level.
- AWS security groups: add inbound rule for the UDP 1194 port on the AWS security group associated with the instance. (Be careful not to delete the SSH rulem otherwise you will lose access to the server!)
- Firewall: 
  - on local machine, run `sudo ufw allow from 10.8.0.1 proto tcp` to open firewall to traffic coming from the instance; 
  - on local machine, run `sudo ufw allow 1194/udp` to open firewall for openvpn; 
  - on AWS instance, run `sudo ufw allow from 10.8.0.6 proto tcp` to open firewall to traffic coming from the local machine.
 You can then run `sudo ufw status verbose` on each machine to check that the right ports are configured.
