import pygame
import sys
from cartpole import CartPole
from neat_algorithm import Population, NeuralNetwork
import visualizer

# Configuration Constants
WIDTH, HEIGHT = 1200, 700
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Cart-Pole Training")
    clock = pygame.time.Clock()

    population = Population(size=10)
    
    # Active agents list containing dictionaries with genome, network, cartpole instance, and alive status
    agents = [] 
    
    def start_generation():
        agents.clear()
        for genome in population.genomes:
            genome.fitness = 0
            net = NeuralNetwork(genome)
            cp = CartPole()
            agents.append({'genome': genome, 'net': net, 'cp': cp, 'alive': True})
    
    start_generation()
    
    generation_max_fitness = 0
    best_genome = None
    
    speed_multiplier = 1
    
    running = True
    while running:
        # Handle external events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    speed_multiplier *= 2
                if event.key == pygame.K_LEFT:
                    speed_multiplier = max(1, speed_multiplier // 2)
        
        # Execute simulation logic, repeating to increase simulation speed
        all_dead = True
        
        for _ in range(speed_multiplier):
            all_dead = True
            for agent in agents:
                if not agent['alive']: continue
                
                all_dead = False
                
                # Neural network inputs: cart position, cart velocity, pole angle, pole angular velocity
                inputs = agent['cp'].state
                output = agent['net'].activate(inputs)
                
                # Determine action based on network output: > 0.5 implies Right (1), otherwise Left (0)
                action = 1 if output[0] > 0.5 else 0
                
                _, _, done = agent['cp'].step(action)
                
                if done:
                    agent['alive'] = False
                else:
                    agent['genome'].fitness += 1
                
                # Cap fitness score to prevent infinite loops once the problem is considered solved
                if agent['genome'].fitness > 2000:
                    agent['alive'] = False
            
            if all_dead:
                break
        
        if all_dead:
            print(f"Generation {population.generation} finished.")
            population.run_generation()
            start_generation()
            continue

        # Identify the finest performing agent that is still alive for visualization purposes
        best_agent = None
        max_fit = -1
        for agent in agents:
            if agent['alive'] and agent['genome'].fitness > max_fit:
                max_fit = agent['genome'].fitness
                best_agent = agent
        
        # If all agents have died in this optimization step, select the first available agent
        if not best_agent and agents:
             best_agent = agents[0]

        # Render simulation and information
        screen.fill(visualizer.BLACK)
        
        if best_agent:
            # Render the simulation visualizer
            visualizer.draw_simulation(screen, best_agent['cp'], 400, 400, scale=100)
            
            # Visual representation of the neural network
            visualizer.draw_network(screen, best_agent['genome'], 750, 50, 400, 400)
            
            # Display simulation statistics and information
            visualizer.draw_text(screen, f"Generation: {population.generation}", 50, 50)
            visualizer.draw_text(screen, f"Alive: {sum(1 for a in agents if a['alive'])} / {len(agents)}", 50, 80)
            visualizer.draw_text(screen, f"Current Fitness: {int(best_agent['genome'].fitness)}", 50, 110)
            visualizer.draw_text(screen, f"Speed: {speed_multiplier}x (Arrows to change)", 50, 140)
            
            if best_agent['genome'].fitness > 1000:
                 visualizer.draw_text(screen, "SOLVED!", 50, 200, size=40, color=visualizer.GREEN)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
