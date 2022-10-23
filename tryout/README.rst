
Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block:: HTML

      <span class="bold>sample HTML</span>


   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/nanodet</span>   
      docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .
      </code></pre>

