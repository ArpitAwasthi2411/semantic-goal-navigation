#!/usr/bin/env python3
"""Type commands for the robot in a second terminal.

    ros2 run semantic_goal_nav chat
    command> take me to the bed

Publishes each line to /user_command and prints the replies from /semantic_goal_nav/feedback.
One-shot:  ros2 run semantic_goal_nav chat "go to the sofa"
"""
import sys
import threading
import time

import rclpy
import rclpy.executors
from rclpy.node import Node
from std_msgs.msg import String


class Chat(Node):
    def __init__(self):
        super().__init__('semantic_goal_nav_chat')
        self.pub = self.create_publisher(String, '/user_command', 10)
        self.create_subscription(String, '/semantic_goal_nav/feedback', self._on_reply, 50)
        self.last_reply = time.time()

    def _on_reply(self, msg):
        self.last_reply = time.time()
        print(f'\r  robot: {msg.data}\ncommand> ', end='', flush=True)

    def send(self, text):
        deadline = time.time() + 3.0
        while self.pub.get_subscription_count() == 0 and time.time() < deadline:
            time.sleep(0.1)
        if self.pub.get_subscription_count() == 0:
            print('  (nobody is listening on /user_command - is the semantic_goal_nav launch running?)')
        self.pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    node = Chat()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        if argv:
            node.send(' '.join(argv))
            t0 = time.time()       # print replies until the robot is quiet for a while
            while time.time() - t0 < 600 and time.time() - node.last_reply < 15:
                time.sleep(0.2)
            return
        print('Commands: "go to the bed", "take me to the kitchen", "go to the chair near the table",\n'
              '"how many chairs are there", "what do you see", "stop".  Ctrl+C to quit.')
        while rclpy.ok():
            text = input('command> ').strip()
            if text:
                node.send(text)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
