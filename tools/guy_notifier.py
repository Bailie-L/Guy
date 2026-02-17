#!/usr/bin/env python3
"""
Guy Communication Notifier - Desktop alerts for new messages
"""
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import os

# Check for required tools
def check_dependencies():
    try:
        subprocess.run(['notify-send', '--version'], capture_output=True, check=True)
    except:
        print("Installing libnotify-bin for desktop notifications...")
        subprocess.run(['sudo', 'apt', 'install', '-y', 'libnotify-bin'])

def send_notification(title, message, urgency="normal", icon="dialog-information"):
    """Send desktop notification"""
    try:
        # Truncate long messages for notification
        if len(message) > 200:
            message = message[:197] + "..."
        
        subprocess.run([
            'notify-send',
            '--urgency', urgency,
            '--icon', icon,
            '--app-name', 'GUY',
            title,
            message
        ])
        
        # Also play a sound if available
        if Path('/usr/share/sounds/freedesktop/stereo/message.oga').exists():
            subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/message.oga'], 
                         capture_output=True)
    except Exception as e:
        print(f"Notification error: {e}")

def monitor_messages():
    """Monitor for new Guy messages"""
    outbox = Path('data/outbox')
    outbox.mkdir(parents=True, exist_ok=True)
    
    # Track seen messages
    seen_messages = set()
    state_file = Path('data/.notifier_seen')
    
    # Load previously seen messages
    if state_file.exists():
        seen_messages = set(state_file.read_text().strip().split('\n'))
    
    # Initial scan
    for msg_file in outbox.glob('msg-*.json'):
        seen_messages.add(msg_file.name)
    
    print(f"🔔 Guy Notifier Active")
    print(f"📂 Monitoring: {outbox.absolute()}")
    print(f"📝 {len(seen_messages)} existing messages ignored")
    print("=" * 50)
    print("Waiting for new messages from Guy...")
    
    try:
        while True:
            # Check for new messages
            for msg_file in sorted(outbox.glob('msg-*.json')):
                if msg_file.name not in seen_messages:
                    try:
                        # Read the message
                        data = json.loads(msg_file.read_text())
                        
                        # Extract content
                        content = data.get('content', 'No content')
                        timestamp = data.get('timestamp', 'Unknown time')
                        msg_type = data.get('type', 'message')
                        
                        # Format timestamp
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = timestamp
                        
                        # Determine urgency based on content
                        urgency = "normal"
                        icon = "dialog-information"
                        
                        if "error" in content.lower() or "fail" in content.lower():
                            urgency = "critical"
                            icon = "dialog-error"
                        elif "warning" in content.lower() or "stuck" in content.lower():
                            urgency = "normal"
                            icon = "dialog-warning"
                        elif "success" in content.lower() or "complete" in content.lower():
                            icon = "dialog-information"
                        
                        # Send notification
                        title = f"🤖 Guy Message [{time_str}]"
                        send_notification(title, content, urgency, icon)
                        
                        # Console output
                        print(f"\n{'='*50}")
                        print(f"⚡ NEW MESSAGE from Guy [{time_str}]")
                        print(f"Type: {msg_type}")
                        print(f"Content: {content[:500]}")
                        print(f"File: {msg_file.name}")
                        print(f"{'='*50}")
                        
                        # Mark as seen
                        seen_messages.add(msg_file.name)
                        
                        # Save state
                        state_file.write_text('\n'.join(seen_messages))
                        
                    except Exception as e:
                        print(f"Error reading {msg_file}: {e}")
                        seen_messages.add(msg_file.name)
            
            # Small delay to prevent CPU spinning
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n✋ Notifier stopped")
        sys.exit(0)

def main():
    # Change to project directory if needed
    project_dir = Path(__file__).parent.parent
    if project_dir.name == 'tools':
        os.chdir(project_dir)
    
    check_dependencies()
    
    # Test notification
    send_notification(
        "🤖 Guy Notifier Started", 
        "You'll receive desktop alerts for new Guy messages",
        "low"
    )
    
    monitor_messages()

if __name__ == "__main__":
    main()
