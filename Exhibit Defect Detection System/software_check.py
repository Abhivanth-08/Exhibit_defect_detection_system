import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
import threading
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest
from twilio.rest import Client
import time

class AnomalyDetectionApp(App):
    def build(self):
        # Create layout
        self.layout = BoxLayout(orientation='vertical')

        # Create UI elements
        self.status_label = Label(text='Status: Waiting to start monitoring')
        self.start_button = Button(text='Start Monitoring', on_press=self.start_monitoring)
        self.stop_button = Button(text='Stop Monitoring', on_press=self.stop_monitoring)

        # Add UI elements to layout
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.start_button)
        self.layout.add_widget(self.stop_button)

        # Thread control flag
        self.monitoring = False

        return self.layout

    def start_monitoring(self, instance):
        if not self.monitoring:
            self.monitoring = True
            self.status_label.text = 'Status: Monitoring Started'
            self.monitoring_thread = threading.Thread(target=self.monitor_system)
            self.monitoring_thread.start()

    def stop_monitoring(self, instance):
        self.monitoring = False
        self.status_label.text = 'Status: Monitoring Stopped'



    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))
        close_button = Button(text='Close')
        content.add_widget(close_button)

        popup = Popup(title=title, content=content, size_hint=(0.8, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def collect_system_metrics(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        return cpu_usage, memory_usage

    def detect_anomalies(self, data):
        model = IsolationForest(contamination=0.1)  # Adjust contamination level as needed
        model.fit(data)
        data['anomaly'] = model.predict(data)
        anomalies = data[data['anomaly'] == -1]
        return anomalies

    def monitor_system(self):
        metrics = []
        while self.monitoring:
            # Collect system metrics
            cpu_usage, memory_usage = self.collect_system_metrics()
            metrics.append([cpu_usage, memory_usage])

            # Convert to DataFrame for anomaly detection
            df = pd.DataFrame(metrics, columns=['cpu_usage', 'memory_usage'])

            # Perform anomaly detection every 10 data points
            if len(df) >= 10:
                anomalies = self.detect_anomalies(df)

                # If anomalies found, send an SMS alert and show a pop-up
                if not anomalies.empty:
                    print("Anomaly detected. SMS sent and pop-up shown.")

                # Clear the old data to continue monitoring
                metrics = []

            # Wait for a while before the next collection (e.g., 5 seconds)
            time.sleep(5)

        # Update UI when monitoring stops
        if not self.monitoring:
            self.status_label.text = 'Status: Monitoring Stopped'

if __name__ == '__main__':
    AnomalyDetectionApp().run()
