import boto3
import sagemaker
from sagemaker import image_uris, get_execution_role
from sagemaker.session import production_variant
from constant import *

def deploy_new_endpoint(model_data,
                        endpoint_name,
                        instance_type='ml.r5.large',
                        instance_count=1,
                        retry_times=0):
    # If an endpoint could describe, it exists, and can not be created by deploy.
    try:
        print(s3_cli.describe_endpoint(EndpointName=endpoint_name))
        return
    except:
        pass

    # edp_model_name = endpoint_name + '-' + str(random.randint(10000, 19999))
    variant_name = "Variant-xlarge-1"  # start from 1, incr 1 when updating.
    img = sagemaker.image_uris.retrieve(
        framework='tensorflow',
        version='1.15',
        region=sm_sess.boto_region_name,
        image_scope='inference',
        instance_type=instance_type
    )

    sm_sess.create_model(
        name=endpoint_name,
        role=role,
        container_defs={
            "Image": img,
            "ModelDataUrl": model_data,
            'Environment': {
                'TF_DISABLE_MKL': '1',
                'TF_DISABLE_POOL_ALLOCATOR': '1',
                # 'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code/',  # Directory inside the container
                # 'SAGEMAKER_PROGRAM': 'inference.py',
            },
        }
    )

    variant1 = production_variant(
        model_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=instance_count,
        variant_name=variant_name,
        initial_weight=1,
    )

    sm_sess.endpoint_from_production_variants(
        name=endpoint_name, production_variants=[variant1],
        tags=[{'Key': 'cost-team', 'Value': 'algorithm'}],
    )
    print(sm_cli.describe_endpoint(EndpointName=endpoint_name))
    # wait_edp_inservice(endpoint_name)


if __name__ == '__main__':
    s3_cli = boto3.client('s3')
    sm_sess = sagemaker.Session()
    print('aws region:', sm_sess.boto_region_name)
    sm_cli = boto3.client('sagemaker')
    role = get_execution_role()
    print('role:', role)
    deploy_new_endpoint(model_data=s3_model_online_tar_file, endpoint_name=endpoint)