# Source: https://github.com/ros2/teleop_twist_keyboard/blob/dashing/teleop_twist_keyboard.py
# Copyright 2011 Brown University Robotics.
# Copyright 2017 Open Source Robotics Foundation, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import threading

import geometry_msgs.msg
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


msg = """
This node takes keypresses from the keyboard and publishes them
as Twist/TwistStamped messages. It works best with a US keyboard layout.
---------------------------
Moving around:
   q    w    e
   a    s    d

For Rotation:
---------------------------
   i    o    p
   j    k    l

anything else : stop

CTRL-C to quit
"""

moveBindings = {
    'w': (0, 0, 1), # forward
    's': (0, 0, -1), # backward
    'a': (-1, 0, 0), # left
    'd': (1, 0, 0), # right
    'q': (0, -1, 0), # up
    'e': (0, 1, 0), # down
}

turnBindings = {
    'p': (1, 0, 0), # pitch up
    'l': (-1, 0, 0), # pitch down
    'o': (0, 0, 1), # roll right
    'k': (0, 0, -1), # roll left
    'i': (0, 1, 0), # yaw right
    'j': (0, -1, 0) # yaw left
}


def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(pos, turn, triangulate=False):
    return 'currently:\tspeed %s\tturn %s\ttriangulate %s' % (pos, turn, triangulate)


def main():
    settings = saveTerminalSettings()

    rclpy.init()

    node = rclpy.create_node('teleop_twist_keyboard')

    # parameters
    TwistMsg = geometry_msgs.msg.Twist

    pub = node.create_publisher(TwistMsg, 'cmd_vel', 10)

    spinner = threading.Thread(target=rclpy.spin, args=(node,))
    spinner.start()

    twist_msg = TwistMsg()
    twist = twist_msg

    try:
        print(msg)
        while True:
            key = getKey(settings)
            triangulate = False
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]

                twist.linear.x = x * 1.0
                twist.linear.y = y * 1.0
                twist.linear.z = z * 1.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0

            elif key in turnBindings.keys():
                x = turnBindings[key][0]
                y = turnBindings[key][1]
                z = turnBindings[key][2]
                twist.angular.x = x * 1.0
                twist.angular.y = y * 1.0
                twist.angular.z = z * 1.0
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                twist.linear.z = 0.0

            elif key == 't':
                twist.linear.x = 100.0
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0
                triangulate = True            
            else:
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0
                if (key == '\x03'):
                    break
            print(
                vels(
                    (twist.linear.x, twist.linear.y, twist.linear.z),
                    (twist.angular.x, twist.angular.y, twist.angular.z),
                    triangulate
                )
            )
            pub.publish(twist_msg)

    except Exception as e:
        print(e)

    finally:
        
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        pub.publish(twist_msg)
        rclpy.shutdown()
        spinner.join()

        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()
