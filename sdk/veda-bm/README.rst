Running Black Marble HD image generation on AWS Elastic Container Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch image generation job
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   from veda_bm import BlackMarbleRunner

   # AWS Access keys to launch containers
   access_key = ''
   secret_key = ''
   # Fetch a token from https://urs.earthdata.nasa.gov/
   earth_data_token = ''

   bm_runner = BlackMarbleRunner(access_key, secret_key, earth_data_token)
   task_arn = bm_runner.run_bm_task('26.718905', '26.520302', '-82.10392', '-81.833752', '2023', '1', '24')
   print('Submitted HD Image generation to task ', task_arn)

Kill an existing job
^^^^^^^^^^^^^^^^^^^^

::

   bm_runner.stop_bm_task(task_arn)
