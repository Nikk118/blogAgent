import axios from "axios"

import { auth } from "@/lib/firebase"

import type { BlogResponse } from "@/types/blog"

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
})

async function getAuthToken() {

  const user = auth.currentUser

  if (!user) {
    throw new Error(
      "User not authenticated"
    )
  }

  return await user.getIdToken(
    true
  )
}

export async function generateBlog(
  topic: string
): Promise<BlogResponse> {

  const token =
    await getAuthToken()

  const response = await API.post(
    "/blog/generate",
    {
      topic,
    },
    {
      headers: {
        Authorization:
          `Bearer ${token}`,
      },
    }
  )

  return response.data
}

export async function getBlogs() {

  const token =
    await getAuthToken()

  const response = await API.get(
    "/blog/all",
    {
      headers: {
        Authorization:
          `Bearer ${token}`,
      },
    }
  )

  return response.data
}

export async function getBlog(
  blogId: string
) {

  const token =
    await getAuthToken()

  const response = await API.get(
    `/blog/${blogId}`,
    {
      headers: {
        Authorization:
          `Bearer ${token}`,
      },
    }
  )

  return response.data
}