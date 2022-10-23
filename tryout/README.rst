
Environment Preparations
------------------------

#. | Build the docker image:

   .. code-block:: HTML

      <span class="bold>sample HTML</span>


   .. raw:: html
      :name:validation

      <code stage="docker_build" style="display:block; white-space:pre-wrap">
      cd <span val="dockerfile_path">hailo_model_zoo/training/nanodet</span>   

      docker build -t nanodet:v0 --build-arg timezone=`cat /etc/timezone` .
      </code>

