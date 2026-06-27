#!/usr/bin/env bash
# pan0.sh — bring up the Bluetooth-PAN bridge for the Dice Tracker.
# Run ON THE PI (invoked automatically by bt-nap.service; safe to run by hand
# with sudo). Idempotent.
#
# Creates a local bridge `pan0` at 192.168.44.1/24 that Bluetooth PAN clients
# (the Surface Pro) attach to. There is NO internet routing / NAT — this is a
# private link between the Pi and the tablet so the phone web UI works over
# Bluetooth with zero WiFi/hotspot. The static IP also means the URL never
# moves (unlike the WiFi DHCP address — see HANDOFF "DHCP moves IPs").
set -euo pipefail

BRIDGE=pan0
ADDR=192.168.44.1/24

# Create the bridge if it doesn't exist yet.
if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
  ip link add name "$BRIDGE" type bridge
fi

# Assign the static IP (replace = idempotent) and bring it up.
ip addr replace "$ADDR" dev "$BRIDGE"
ip link set "$BRIDGE" up

echo "pan0 up: $ADDR"
