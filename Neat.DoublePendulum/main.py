import pygame
import sys
from double_pendulum import DoublePendulum
from neat_algorithm import Population, NeuralNetwork
import visualizer

# Configuration Constants
WIDTH, HEIGHT = 1200, 700
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Double Pendulum Training")
    clock = pygame.time.Clock()

    population = Population(size=10)
    
    # Active agents list containing dictionaries with genome, network, simulation instance, and alive status
    agents = [] 
    
    def start_generation():
        agents.clear()
        for genome in population.genomes:
            genome.fitness = 0
            net = NeuralNetwork(genome)
            dp = DoublePendulum()
            agents.append({'genome': genome, 'net': net, 'dp': dp, 'alive': True})
    
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
                
                # Neural network inputs: Position, Velocity, Pole 1 Angle/Vel, Pole 2 Angle/Vel
                inputs = agent['dp'].state
                output = agent['net'].activate(inputs)
                
                # Determine action based on network output
                action = 1 if output[0] > 0.5 else 0
                
                _, reward, done = agent['dp'].step(action)
                
                if done:
                    agent['alive'] = False
                    # Penalize dying early if needed, but reward is already 0
                else:
                    agent['genome'].fitness += reward
                
                # Cap fitness score to prevent infinite loops once the problem is considered solved
                # Maximum estimated fitness.
                if agent['genome'].fitness > 10000: 
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
            visualizer.draw_simulation(screen, best_agent['dp'], 400, 400, scale=100)
            
            # Visual representation of the neural network
            visualizer.draw_network(screen, best_agent['genome'], 750, 50, 400, 400)
            
            # Display simulation statistics and information
            visualizer.draw_text(screen, f"Generation: {population.generation}", 50, 50)
            visualizer.draw_text(screen, f"Alive: {sum(1 for a in agents if a['alive'])} / {len(agents)}", 50, 80)
            visualizer.draw_text(screen, f"Current Fitness: {int(best_agent['genome'].fitness)}", 50, 110)
            visualizer.draw_text(screen, f"Speed: {speed_multiplier}x (Arrows to change)", 50, 140)
            
            if best_agent['genome'].fitness > 1000:
                 visualizer.draw_text(screen, "SOLVED?", 50, 200, size=40, color=visualizer.GREEN)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
