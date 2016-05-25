#include <random>

class generator_store{
 public:
  generator_store(int num_threads,int stride);
 ~generator_store();
  
  void reseed(int thread, int nps);
  
  double sample(int thread);

 private:
  std::vector<std::default_random_engine> generators;
  int stride;
};
