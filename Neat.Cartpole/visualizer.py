import pygame
import math

# Color Definitions
WHITE = (255, 255, 255)
BLACK = (20, 20, 25)
GRAY = (200, 200, 200)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 180, 255)
CART_COLOR = (50, 50, 50)
POLE_COLOR = (200, 150, 100)

def draw_text(screen, text, x, y, size=20, color=WHITE):
    font = pygame.font.SysFont("Arial", size)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def draw_simulation(screen, cart_pole, x_offset, y_offset, scale=100):
    x, x_dot, theta, theta_dot = cart_pole.state
    
    # Render Track
    pygame.draw.line(screen, GRAY, (x_offset - 2.4 * scale, y_offset), (x_offset + 2.4 * scale, y_offset), 2)
    
    # Render Cart
    cart_x = x_offset + x * scale
    cart_y = y_offset
    cart_width = 50
    cart_height = 30
    
    pygame.draw.rect(screen, CART_COLOR, (cart_x - cart_width/2, cart_y - cart_height/2, cart_width, cart_height))
    
    # Render Pole
    pole_len = cart_pole.length * 2 * scale
    pole_end_x = cart_x + math.sin(theta) * pole_len
    pole_end_y = cart_y - math.cos(theta) * pole_len
    
    pygame.draw.line(screen, POLE_COLOR, (cart_x, cart_y), (pole_end_x, pole_end_y), 6)
    
    # Render Joints
    pygame.draw.circle(screen, RED, (int(cart_x), int(cart_y)), 5)

def draw_network(screen, genome, x, y, width, height):
    # Render background container
    pygame.draw.rect(screen, (30, 30, 40), (x, y, width, height), border_radius=10)
    pygame.draw.rect(screen, (100, 100, 120), (x, y, width, height), 2, border_radius=10)
    
    if not genome:
        return

    # Calculate and map node positions
    node_pos = {}
    
    # Organize nodes by their layer index
    layers = {} # layer_index -> list of nodes
    for node in genome.nodes:
        l = node.layer
        if l not in layers: layers[l] = []
        layers[l].append(node)
        
    layer_keys = sorted(layers.keys())
    
    # Calculate positions
    for i, l_key in enumerate(layer_keys):
        nodes_in_layer = layers[l_key]
        layer_x = x + 50 + (i / (len(layer_keys) - 1)) * (width - 100) if len(layer_keys) > 1 else x + width/2
        
        for j, node in enumerate(nodes_in_layer):
            node_y = y + 50 + (j / (len(nodes_in_layer))) * (height - 100)
            # Center vertically if few nodes
            if len(nodes_in_layer) == 1: node_y = y + height/2
            
            node_pos[node.id] = (layer_x, node_y)

    # Render Neural Connections
    for conn in genome.connections:
        if not conn.enabled: continue
        if conn.in_node not in node_pos or conn.out_node not in node_pos: continue
        
        start = node_pos[conn.in_node]
        end = node_pos[conn.out_node]
        
        color = GREEN if conn.weight > 0 else RED
        thickness = max(1, min(5, int(abs(conn.weight) * 2)))
        
        pygame.draw.line(screen, color, start, end, thickness)

    # Render Neural Nodes
    font = pygame.font.SysFont("Arial", 12)
    
    for node in genome.nodes:
        if node.id not in node_pos: continue
        px, py = node_pos[node.id]
        
        color = WHITE
        label = ""
        
        if node.type == 'INPUT': 
            color = BLUE
            if node.id == 0: label = "x"
            elif node.id == 1: label = "dx"
            elif node.id == 2: label = "th"
            elif node.id == 3: label = "dth"
        elif node.type == 'OUTPUT': 
            color = (255, 255, 0)
            if node.id == 4: label = "F"
        
        # Draw node circle
        pygame.draw.circle(screen, color, (int(px), int(py)), 10)
        
        # Draw Label
        if label:
            text_surf = font.render(label, True, BLACK)
            text_rect = text_surf.get_rect(center=(int(px), int(py)))
            screen.blit(text_surf, text_rect)
        elif node.type == 'HIDDEN':
             pass


        # Implement hover functionality to display details
        mx, my = pygame.mouse.get_pos()
        dist = math.hypot(mx - px, my - py)
        if dist < 10:
            desc = f"ID: {node.id} ({node.type})"
            if node.type == 'INPUT': 
                if node.id == 0: desc = "Cart Position"
                elif node.id == 1: desc = "Cart Velocity"
                elif node.id == 2: desc = "Pole Angle"
                elif node.id == 3: desc = "Pole Angular Velocity"
            elif node.type == 'OUTPUT': 
                if node.id == 4: desc = "Force"
            
            # Render descriptive tooltip
            tip_font = pygame.font.SysFont("Arial", 16)
            tip_surf = tip_font.render(desc, True, WHITE)
            tip_bg = pygame.Rect(mx + 10, my + 10, tip_surf.get_width() + 10, tip_surf.get_height() + 10)
            
            pygame.draw.rect(screen, (50, 50, 50), tip_bg, border_radius=5)
            pygame.draw.rect(screen, WHITE, tip_bg, 1, border_radius=5)
            screen.blit(tip_surf, (mx + 15, my + 15))


