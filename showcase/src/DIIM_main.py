import DIIM_init as init
import DIIM_config as config


#import DIIM_LSI_Similarity as lsi_sim # this is old way maybe
import DIIM_word2vec as diim_w2v

logger = config.logger

def diim_main():
    #init.diim_init_all()
    #lsi_sim.build_LSI_Models_of_Ontos_and_IoT()
    diim_w2v.diim_execution_routine()

diim_main()